#!/usr/bin/env python3
"""CSP pilot: can CAWR-biased MatterSim relaxation find gamma-B28?

Protocol (matched budget, per the CRISP audit discipline):
  1. Generate N random 28-atom boron structures with pyxtal (random
     space groups, fixed seed).
  2. Relax each under P = 50 GPa (gamma-B28 stability field) with TWO
     arms at the SAME 200-step budget:
       control   — plain MatterSim-1M, FIRE + FrechetCellFilter
       cawr-bias — MixedCalculator(mode='bias'): E/stress from MatterSim,
                   F = F_ms + lambda(t)*F_cawr, lambda annealed to 0 over
                   the 200 steps (adaptive eta=0.3); CAWR labels refresh
                   with static exhaustion every 10 steps.
  3. Score: spglib SG (symprec sweep), enthalpy/atom vs the gamma-B28
     reference relaxed under the same potential+pressure, Hungarian fp
     distance to the relaxed reference.

Success per structure: SG = Pnnm (#58) at any swept symprec AND
enthalpy within 5 meV/atom of the reference.

The reference structure ships in the repo (benchmark/gamma_b28_Pnnm.vasp);
override with --ref-poscar. The output JSON records the reference path +
sha256, all parameters, and an explicit `acceptance` verdict block.

Usage:
  python benchmark/cawr_mattersim_csp.py --n 50 --steps 200 \
      --ref-poscar benchmark/gamma_b28_Pnnm.vasp --out benchmark/csp_results.json
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import ase.io
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.units import GPa
import spglib
from scipy.optimize import linear_sum_assignment

P_GPA = 50.0
# Repo-controlled reference (28-atom gamma-B28, Pnnm #58); override with
# --ref-poscar. Path is relative to this script so a clean checkout works.
DEFAULT_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gamma_b28_Pnnm.vasp')
SYMPREC_SWEEP = (1e-3, 1e-2, 3e-2, 5e-2, 1e-1)
H_TOL = 0.005           # eV/atom window around the reference enthalpy
FP_CUTOFF, FP_NX = 4.0, 128
TARGET_SG = 'Pnnm'      # gamma-B28 space group (#58)


# ── helpers ──────────────────────────────────────────────────────────

def make_mattersim():
    from mattersim.forcefield import MatterSimCalculator
    return MatterSimCalculator(device="cpu")


def gen_structures(n, seed):
    """N pyxtal random crystals: 28 B atoms, random space group."""
    from pyxtal import pyxtal
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        sg = int(rng.integers(2, 231))
        x = pyxtal()
        try:
            x.from_random(3, sg, ['B'], [28], factor=0.85,
                          random_state=int(rng.integers(0, 2 ** 31)))
            atoms = x.to_ase()
        except Exception:
            continue
        if len(atoms) == 28:
            atoms.set_pbc(True)
            out.append({'gen_sg': sg, 'atoms': atoms})
    return out


def sg_sweep(atoms):
    """Spacegroup across the symprec sweep; returns {symprec: 'SG (#)'}"""
    res = {}
    cell = (atoms.cell[:], atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    for sp in SYMPREC_SWEEP:
        try:
            res[sp] = spglib.get_spacegroup(cell, symprec=sp) or 'P1 (1)'
        except Exception:
            res[sp] = 'ERR'
    return res


def enthalpy_per_atom(atoms, energy):
    return (energy + P_GPA * GPa * atoms.get_volume()) / len(atoms)


def fp_dist(atoms1, atoms2):
    """Hungarian-matched per-atom fingerprint distance (drive metric,
    not an identity metric)."""
    from reformpy.cawr import compute_fp
    fp1 = compute_fp(atoms1, backend='torch', cutoff=FP_CUTOFF, nx=FP_NX)
    fp2 = compute_fp(atoms2, backend='torch', cutoff=FP_CUTOFF, nx=FP_NX)
    nat = len(fp1)
    cost = np.linalg.norm(fp1[:, None, :] - fp2[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum() / nat)


# ── relaxation arms (matched budget) ─────────────────────────────────

def relax_control(atoms, steps, fmax=0.02):
    a = atoms.copy()
    a.calc = make_mattersim()
    opt = FIRE(FrechetCellFilter(a, scalar_pressure=P_GPA * GPa),
               logfile=None)
    opt.run(fmax=fmax, steps=steps)
    a.calc = make_mattersim()  # fresh, to read converged E cleanly
    return a, opt.nsteps


def relax_cawr_bias(atoms, steps, fmax=0.02, refresh_every=10):
    from reformpy.mixing import MixedCalculator
    from reformpy.cawr_calculator import CAWRCalculator
    a = atoms.copy()
    cawr = CAWRCalculator(cutoff=FP_CUTOFF, nx=FP_NX, backend='torch')
    mixed = MixedCalculator(make_mattersim(), cawr, iter_max=steps,
                            scheme='cosine', mode='bias',
                            adaptive_lambda=True, eta=0.3)
    a.calc = mixed
    opt = FIRE(FrechetCellFilter(a, scalar_pressure=P_GPA * GPa),
               logfile=None)

    def refresh():
        # round boundary every `refresh_every` steps: exhaust statically
        # justified proposals (mirrors cawr_reform's discovery loop)
        if opt.nsteps == 0 or opt.nsteps % refresh_every:
            return
        committed = False
        for _ in range(50):
            cawr.refresh_labels(a)
            st = cawr.state
            if st.last_committed:
                committed = True
            if not (st.has_pending or st.last_committed):
                break
        if committed:
            # A label commit changes the CAWR force law WITHOUT moving atoms.
            # CAWRCalculator.refresh_labels clears its own cache, but the
            # MixedCalculator wrapper caches the COMBINED result keyed on
            # atomic state and would serve stale forces at this geometry.
            # Invalidate the wrapper too so the next evaluation recomputes.
            mixed.reset()

    opt.attach(refresh)
    opt.run(fmax=fmax, steps=steps)
    K = cawr.state.K_per_element() if cawr.state is not None else {}
    a.calc = make_mattersim()
    return a, opt.nsteps, K


# ── scoring ──────────────────────────────────────────────────────────

def score(atoms, h_ref, ref_relaxed):
    e = atoms.get_potential_energy()
    h = enthalpy_per_atom(atoms, e)
    sgs = sg_sweep(atoms)
    found_target = any(TARGET_SG in s for s in sgs.values())
    d_fp = fp_dist(atoms, ref_relaxed)
    success = found_target and abs(h - h_ref) < H_TOL
    return {'H_per_atom': h, 'dH_meV': (h - h_ref) * 1000.0,
            'sg_sweep': {str(k): v for k, v in sgs.items()},
            'target_sg_found': bool(found_target), 'fp_dist_to_ref': d_fp,
            'success': bool(success)}


def build_acceptance(results, h_ref, ref_sg_ok):
    """Explicit, machine-readable pass/fail gate (F3).

    Success criterion (per structure): target SG found AND |dH| < H_TOL.
    Arm verdict: an arm 'recovers' gamma-B28 if it has >=1 success.
    Overall verdict compares the CAWR-bias arm to the control arm.
    """
    n = len(results)
    ok_c = sum(r['control']['success'] for r in results)
    ok_b = sum(r['cawr_bias']['success'] for r in results)
    tgt_c = sum(r['control']['target_sg_found'] for r in results)
    tgt_b = sum(r['cawr_bias']['target_sg_found'] for r in results)
    dh_c = [r['control']['dH_meV'] for r in results]
    dh_b = [r['cawr_bias']['dH_meV'] for r in results]
    better = int(sum(b < c - 1.0 for b, c in zip(dh_b, dh_c)))
    worse = int(sum(b > c + 1.0 for b, c in zip(dh_b, dh_c)))
    if ok_b > ok_c:
        verdict = 'cawr_bias_wins'
    elif ok_b < ok_c:
        verdict = 'control_wins'
    elif ok_b == 0 and ok_c == 0:
        verdict = 'both_failed_no_recovery'
    else:
        verdict = 'tie'
    return {
        'success_criterion': (f"space group contains '{TARGET_SG}' at any "
                              f"swept symprec AND |dH| < {H_TOL * 1000:.0f} "
                              f"meV/atom vs the relaxed reference"),
        'reference_relaxed_to_target_sg': bool(ref_sg_ok),
        'n_structures': n,
        'control': {'success': ok_c, 'target_sg_found': tgt_c,
                    'dH_median_meV': float(np.median(dh_c)) if n else None,
                    'dH_min_meV': float(np.min(dh_c)) if n else None},
        'cawr_bias': {'success': ok_b, 'target_sg_found': tgt_b,
                      'dH_median_meV': float(np.median(dh_b)) if n else None,
                      'dH_min_meV': float(np.min(dh_b)) if n else None},
        'head_to_head_dH': {'bias_better': better, 'bias_worse': worse,
                            'tie': n - better - worse},
        'verdict': verdict,
        'caveat': ('A fixed-budget bias-vs-control comparison, NOT a CSP '
                   'capability claim: gamma-B28 needs far more search effort '
                   'than N x steps of local relaxation. Do not report a '
                   'success claim unless verdict == cawr_bias_wins.'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ref-poscar', default=DEFAULT_REF,
                   help='Reference gamma-B28 structure (default: repo copy)')
    p.add_argument('--out', default='benchmark/csp_results.json')
    p.add_argument('--structures-out', default='benchmark/csp_structures.xyz')
    args = p.parse_args()

    if not os.path.isfile(args.ref_poscar):
        raise SystemExit(f"reference structure not found: {args.ref_poscar}\n"
                         f"pass --ref-poscar <path> to a gamma-B28 POSCAR.")
    ref_sha = hashlib.sha256(open(args.ref_poscar, 'rb').read()).hexdigest()

    t_start = time.time()

    # Reference: relax gamma-B28 under the same potential + pressure
    # (double budget so the reference itself is well converged).
    ref = ase.io.read(args.ref_poscar)
    ref_relaxed, _ = relax_control(ref, steps=2 * args.steps, fmax=0.01)
    h_ref = enthalpy_per_atom(ref_relaxed, ref_relaxed.get_potential_energy())
    ref_sweep = sg_sweep(ref_relaxed)
    ref_sg_ok = any(TARGET_SG in s for s in ref_sweep.values())
    print(f"reference {os.path.basename(args.ref_poscar)} @ {P_GPA} GPa: "
          f"H = {h_ref:.4f} eV/at, SG sweep: {ref_sweep}", flush=True)
    if not ref_sg_ok:
        print(f"  WARNING: relaxed reference is not {TARGET_SG} under this "
              f"potential+pressure; scoring baseline is suspect.", flush=True)

    provenance = {'h_ref': h_ref, 'P_GPa': P_GPA, 'steps': args.steps,
                  'seed': args.seed, 'n_requested': args.n,
                  'ref_poscar': os.path.abspath(args.ref_poscar),
                  'ref_sha256': ref_sha, 'target_sg': TARGET_SG,
                  'H_tol_eV_per_atom': H_TOL,
                  'fp_cutoff': FP_CUTOFF, 'fp_nx': FP_NX}

    print(f"Generating {args.n} pyxtal structures (seed {args.seed})...",
          flush=True)
    pool = gen_structures(args.n, args.seed)
    print(f"  got {len(pool)}", flush=True)

    def _json_default(o):
        if hasattr(o, 'item'):       # numpy scalars (int64/float64/bool_)
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")

    def save(results, done):
        payload = dict(provenance)
        payload['complete'] = done
        payload['results'] = results
        if done:
            payload['acceptance'] = build_acceptance(results, h_ref, ref_sg_ok)
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=1, default=_json_default)

    results = []
    for i, entry in enumerate(pool):
        row = {'idx': i, 'gen_sg': entry['gen_sg']}
        t0 = time.time()

        ctrl, n1 = relax_control(entry['atoms'], args.steps)
        row['control'] = score(ctrl, h_ref, ref_relaxed)
        row['control']['nsteps'] = int(n1)

        bias, n2, K = relax_cawr_bias(entry['atoms'], args.steps)
        row['cawr_bias'] = score(bias, h_ref, ref_relaxed)
        row['cawr_bias']['nsteps'] = int(n2)
        row['cawr_bias']['K_final'] = {str(k): v for k, v in K.items()}

        row['wall_s'] = round(time.time() - t0, 1)
        results.append(row)

        ctrl.info.update(idx=i, arm='control')
        bias.info.update(idx=i, arm='cawr_bias')
        ase.io.write(args.structures_out, [ctrl, bias], append=(i > 0))

        print(f"[{i + 1}/{len(pool)}] gen_sg={entry['gen_sg']:3d}  "
              f"ctrl: dH={row['control']['dH_meV']:8.1f} meV "
              f"sg={row['control']['target_sg_found']} "
              f"S={row['control']['success']}  |  "
              f"bias: dH={row['cawr_bias']['dH_meV']:8.1f} meV "
              f"sg={row['cawr_bias']['target_sg_found']} "
              f"S={row['cawr_bias']['success']} "
              f"K={row['cawr_bias']['K_final']}  ({row['wall_s']}s)", flush=True)

        save(results, done=False)  # incremental checkpoint

    save(results, done=True)
    acc = build_acceptance(results, h_ref, ref_sg_ok)
    print(f"\n=== SUMMARY (N={len(results)}, {args.steps} steps, "
          f"{P_GPA} GPa, {time.time() - t_start:.0f}s) ===")
    print(f"control  : success {acc['control']['success']}/{len(results)}  "
          f"{TARGET_SG} found {acc['control']['target_sg_found']}  "
          f"dH_median {acc['control']['dH_median_meV']:.1f} meV")
    print(f"cawr-bias: success {acc['cawr_bias']['success']}/{len(results)}  "
          f"{TARGET_SG} found {acc['cawr_bias']['target_sg_found']}  "
          f"dH_median {acc['cawr_bias']['dH_median_meV']:.1f} meV")
    print(f"head-to-head dH: {acc['head_to_head_dH']}")
    print(f"VERDICT: {acc['verdict']}")


if __name__ == '__main__':
    main()
