#!/usr/bin/env python3
"""CAWR CSP campaign: 3-arm comparison across several test systems.

Arms (all matched budget, FIRE fmax/steps, FrechetCellFilter + pressure):
  normal  -- plain MatterSim
  reform  -- MatterSim + fingerprint bias, K frozen at 1 (single-environment
             reform: drive all same-element atoms to ONE environment)
  cawr    -- MatterSim + bias, annealed-K multi-environment discovery

Modes:
  gen        (local)  N pyxtal structures per system -> <pool>/<sys>.xyz
  reference  (cluster) relax the known ground state  -> <refs>/<sys>.json
  run        (cluster) relax ONE structure, 3 arms   -> <out>/<sys>_<idx>.json
  aggregate  (local)  combine result rows            -> summary JSON

Designed for SLURM 1-core array jobs (run mode). The job script must set
`ulimit -l unlimited` so the (unused) MPI auto-init from reformpy.calculator
succeeds on compute nodes; this campaign never uses MPI.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# system -> generation + scoring config. `counts` must sum to the reference
# atom count so random pools match the reference stoichiometry/cell size.
SYSTEMS = {
    'B28':  dict(pressure=50.0, target_sg='Pnnm',     species=['B'],
                 counts=[28], factor=0.85, ref='file:gamma_b28_Pnnm.vasp',
                 role='multi-orbit flagship (8 B orbits)'),
    'MgO':  dict(pressure=0.0,  target_sg='Fm-3m',    species=['Mg', 'O'],
                 counts=[4, 4], factor=1.10, ref='builtin:MgO',
                 role='ionic single-orbit sanity'),
    'TiO2': dict(pressure=0.0,  target_sg='P4_2/mnm', species=['Ti', 'O'],
                 counts=[2, 4], factor=1.10, ref='builtin:TiO2',
                 role='binary single-orbit sanity'),
    'Si':   dict(pressure=0.0,  target_sg='Fd-3m',    species=['Si'],
                 counts=[8], factor=1.10, ref='builtin:Si',
                 role='covalent elemental sanity'),
}
SYMPREC_SWEEP = (1e-3, 1e-2, 3e-2, 5e-2, 1e-1)
H_TOL = 0.005            # eV/atom enthalpy window for "success"
FP_CUTOFF, FP_NX = 4.0, 128
FMAX, STEPS = 0.005, 2000
ETA = 0.3                # adaptive-lambda force-scale ratio
# arms:
#   normal       plain MatterSim
#   reform/cawr  MatterSim + fingerprint bias (K=1 / annealed-K), stress mixed
#                so the fingerprint drives the cell too (mix_stress=True)
#   *_pure       potential-free fingerprint relaxation (torch forces+stress),
#                then scored by a MatterSim single point on the relaxed cell
ARMS = ('normal', 'reform', 'cawr', 'reform_pure', 'cawr_pure')


# ── shared helpers ───────────────────────────────────────────────────

def make_mattersim():
    from mattersim.forcefield import MatterSimCalculator
    return MatterSimCalculator(device='cpu')


def build_reference(sysname):
    """The known ground-state structure (unrelaxed)."""
    spec = SYSTEMS[sysname]['ref']
    kind, val = spec.split(':', 1)
    if kind == 'file':
        import ase.io
        return ase.io.read(os.path.join(HERE, val))
    import ase.build
    from ase.spacegroup import crystal
    if val == 'MgO':
        return ase.build.bulk('MgO', 'rocksalt', a=4.21, cubic=True)
    if val == 'Si':
        return ase.build.bulk('Si', 'diamond', a=5.43, cubic=True)
    if val == 'TiO2':                                   # rutile P4_2/mnm
        return crystal(['Ti', 'O'], [(0, 0, 0), (0.3053, 0.3053, 0.0)],
                       spacegroup=136,
                       cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
    raise ValueError(f"unknown builtin reference {val}")


def sg_sweep(atoms):
    import spglib
    cell = (atoms.cell[:], atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    out = {}
    for sp in SYMPREC_SWEEP:
        try:
            out[str(sp)] = spglib.get_spacegroup(cell, symprec=sp) or 'P1 (1)'
        except Exception:
            out[str(sp)] = 'ERR'
    return out


def enthalpy_per_atom(atoms, energy, pressure):
    from ase.units import GPa
    return (energy + pressure * GPa * atoms.get_volume()) / len(atoms)


def fp_dist(atoms1, atoms2):
    """Hungarian-matched per-atom fingerprint distance (drive metric)."""
    from reformpy.cawr import compute_fp
    from scipy.optimize import linear_sum_assignment
    f1 = compute_fp(atoms1, backend='torch', cutoff=FP_CUTOFF, nx=FP_NX)
    f2 = compute_fp(atoms2, backend='torch', cutoff=FP_CUTOFF, nx=FP_NX)
    if f1.shape != f2.shape:
        return float('nan')
    cost = np.linalg.norm(f1[:, None, :] - f2[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].sum() / len(f1))


# ── relaxation arms ──────────────────────────────────────────────────

def relax(atoms, arm, pressure, steps=STEPS, fmax=FMAX, refresh_every=10):
    """Return (relaxed_atoms, nsteps, K_per_element)."""
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from ase.units import GPa
    a = atoms.copy()

    if arm == 'normal':
        a.calc = make_mattersim()
        opt = FIRE(FrechetCellFilter(a, scalar_pressure=pressure * GPa),
                   logfile=None)
        opt.run(fmax=fmax, steps=steps)
        a.calc = make_mattersim()
        return a, int(opt.nsteps), {}

    from reformpy.cawr_calculator import CAWRCalculator
    cawr = CAWRCalculator(cutoff=FP_CUTOFF, nx=FP_NX, backend='torch')
    annealed = arm in ('cawr', 'cawr_pure')
    pure = arm.endswith('_pure')

    if pure:
        # potential-free: fingerprint drives positions AND cell via torch
        # autograd forces+stress. Scored later by a MatterSim single point.
        a.calc = cawr
    else:
        from reformpy.mixing import MixedCalculator
        mixed = MixedCalculator(make_mattersim(), cawr, iter_max=steps,
                                scheme='cosine', mode='bias',
                                adaptive_lambda=True, eta=ETA, mix_stress=True)
        a.calc = mixed
    opt = FIRE(FrechetCellFilter(a, scalar_pressure=pressure * GPa),
               logfile=None)

    if annealed:
        # annealed-K discovery: at each round boundary exhaust statically
        # justified split/merge proposals. refresh_labels clears the CAWR
        # cache on commit; for the bias arm also reset the wrapper cache.
        def refresh():
            if opt.nsteps == 0 or opt.nsteps % refresh_every:
                return
            committed = False
            for _ in range(50):
                cawr.refresh_labels(a)
                if cawr.state.last_committed:
                    committed = True
                if not (cawr.state.has_pending or cawr.state.last_committed):
                    break
            if committed and not pure:
                mixed.reset()
        opt.attach(refresh)
    # reform / reform_pure: no refresh -> K stays 1 (single-environment)

    opt.run(fmax=fmax, steps=steps)
    K = cawr.state.K_per_element() if cawr.state is not None else {}
    a.calc = make_mattersim()                     # physical scoring
    return a, int(opt.nsteps), K


# ── modes ────────────────────────────────────────────────────────────

def mode_gen(args):
    """Generate N pyxtal random structures per system (run locally)."""
    from pyxtal import pyxtal
    import ase.io
    os.makedirs(args.pool, exist_ok=True)
    systems = args.systems or list(SYSTEMS)
    for sysname in systems:
        s = SYSTEMS[sysname]
        rng = np.random.default_rng(args.seed)
        frames, attempts = [], 0
        while len(frames) < args.n and attempts < args.n * 80:
            attempts += 1
            sg = int(rng.integers(2, 231))
            x = pyxtal()
            try:
                x.from_random(3, sg, s['species'], s['counts'],
                              factor=s['factor'],
                              random_state=int(rng.integers(0, 2 ** 31)))
                atoms = x.to_ase()
            except Exception:
                continue
            if len(atoms) == sum(s['counts']):
                atoms.set_pbc(True)
                atoms.info['gen_sg'] = sg
                frames.append(atoms)
        path = os.path.join(args.pool, f"{sysname}.xyz")
        ase.io.write(path, frames)
        print(f"{sysname}: wrote {len(frames)} structures -> {path} "
              f"({attempts} attempts)", flush=True)


def mode_reference(args):
    """Relax the known ground state under MatterSim + pressure (on cluster)."""
    import ase.io
    os.makedirs(args.refs, exist_ok=True)
    s = SYSTEMS[args.system]
    ref = build_reference(args.system)
    relaxed, nsteps, _ = relax(ref, 'normal', s['pressure'],
                               steps=2 * args.steps, fmax=0.001)
    h = enthalpy_per_atom(relaxed, relaxed.get_potential_energy(),
                          s['pressure'])
    sweep = sg_sweep(relaxed)
    ok = any(s['target_sg'] in v for v in sweep.values())
    out = os.path.join(args.refs, f"{args.system}.json")
    ase.io.write(os.path.join(args.refs, f"{args.system}_ref.xyz"), relaxed)
    with open(out, 'w') as f:
        json.dump({'system': args.system, 'h_ref': h, 'pressure': s['pressure'],
                   'target_sg': s['target_sg'], 'sg_sweep': sweep,
                   'ref_is_target_sg': bool(ok), 'nsteps': nsteps}, f, indent=1)
    print(f"{args.system}: H_ref={h:.5f} eV/at  target_sg_ok={ok}  "
          f"sweep={sweep}  -> {out}", flush=True)


def _score(atoms, h_ref, pressure, target_sg, ref_relaxed):
    e = atoms.get_potential_energy()
    h = enthalpy_per_atom(atoms, e, pressure)
    sweep = sg_sweep(atoms)
    found = any(target_sg in v for v in sweep.values())
    return {'H_per_atom': h, 'dH_meV': (h - h_ref) * 1000.0,
            'sg_sweep': sweep, 'target_sg_found': bool(found),
            'fp_dist_to_ref': fp_dist(atoms, ref_relaxed),
            'success': bool(found and abs(h - h_ref) < H_TOL)}


def mode_run(args):
    """Relax ONE pooled structure through all 3 arms (one SLURM array task)."""
    import ase.io
    os.makedirs(args.out, exist_ok=True)
    s = SYSTEMS[args.system]
    pool = ase.io.read(os.path.join(args.pool, f"{args.system}.xyz"), index=':')
    if args.idx >= len(pool):
        raise SystemExit(f"idx {args.idx} >= pool size {len(pool)}")
    atoms = pool[args.idx]
    refj = json.load(open(os.path.join(args.refs, f"{args.system}.json")))
    h_ref = refj['h_ref']
    ref_relaxed = ase.io.read(os.path.join(args.refs, f"{args.system}_ref.xyz"))

    row = {'system': args.system, 'idx': args.idx,
           'gen_sg': int(atoms.info.get('gen_sg', -1))}
    t0 = time.time()
    for arm in ARMS:
        relaxed, nsteps, K = relax(atoms, arm, s['pressure'],
                                   steps=args.steps, fmax=args.fmax)
        sc = _score(relaxed, h_ref, s['pressure'], s['target_sg'], ref_relaxed)
        sc['nsteps'] = nsteps
        sc['K_final'] = {str(k): v for k, v in K.items()}
        row[arm] = sc
    row['wall_s'] = round(time.time() - t0, 1)

    out = os.path.join(args.out, f"{args.system}_{args.idx:04d}.json")
    with open(out, 'w') as f:
        json.dump(row, f, indent=1, default=lambda o: o.item()
                  if hasattr(o, 'item') else str(o))
    print(f"[{args.system} #{args.idx}] "
          + "  ".join(f"{a}: dH={row[a]['dH_meV']:.0f} S={row[a]['success']}"
                      for a in ARMS)
          + f"  K={row['cawr']['K_final']}  ({row['wall_s']}s)", flush=True)


def mode_aggregate(args):
    """Combine per-structure result JSONs into a summary with acceptance."""
    rows = [json.load(open(p)) for p in sorted(glob.glob(
        os.path.join(args.out, '*_[0-9]*.json')))]
    by_sys = {}
    for r in rows:
        by_sys.setdefault(r['system'], []).append(r)
    summary = {}
    for sysname, rs in sorted(by_sys.items()):
        arms = {}
        for arm in ARMS:
            ok = sum(r[arm]['success'] for r in rs)
            tgt = sum(r[arm]['target_sg_found'] for r in rs)
            dh = [r[arm]['dH_meV'] for r in rs]
            arms[arm] = {'success': ok, 'target_sg_found': tgt,
                         'dH_median_meV': float(np.median(dh)),
                         'dH_min_meV': float(np.min(dh))}
        # which arm reaches the lowest dH per structure
        wins = {a: 0 for a in ARMS}
        for r in rs:
            best = min(ARMS, key=lambda a: r[a]['dH_meV'])
            wins[best] += 1
        summary[sysname] = {'n': len(rs), 'role': SYSTEMS[sysname]['role'],
                            'arms': arms, 'lowest_dH_arm_counts': wins}
    out = {'config': {'steps': STEPS, 'fmax': FMAX, 'H_tol_eV': H_TOL,
                      'fp_cutoff': FP_CUTOFF, 'fp_nx': FP_NX, 'eta': ETA},
           'systems': summary,
           'caveat': ('Fixed-budget 3-arm comparison; report a CAWR benefit '
                      'only where its success/lowest-dH counts beat BOTH '
                      'reform and normal on the multi-orbit system.')}
    with open(args.summary, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out['systems'], indent=1))
    print(f"\nwrote {args.summary}  ({len(rows)} structures)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)

    g = sub.add_parser('gen'); g.set_defaults(fn=mode_gen)
    g.add_argument('--n', type=int, default=300)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--pool', default='pools')
    g.add_argument('--systems', nargs='*')

    r = sub.add_parser('reference'); r.set_defaults(fn=mode_reference)
    r.add_argument('--system', required=True, choices=list(SYSTEMS))
    r.add_argument('--refs', default='refs')
    r.add_argument('--steps', type=int, default=STEPS)

    u = sub.add_parser('run'); u.set_defaults(fn=mode_run)
    u.add_argument('--system', required=True, choices=list(SYSTEMS))
    u.add_argument('--idx', type=int, required=True)
    u.add_argument('--pool', default='pools')
    u.add_argument('--refs', default='refs')
    u.add_argument('--out', default='results')
    u.add_argument('--steps', type=int, default=STEPS)
    u.add_argument('--fmax', type=float, default=FMAX)

    a = sub.add_parser('aggregate'); a.set_defaults(fn=mode_aggregate)
    a.add_argument('--out', default='results')
    a.add_argument('--summary', default='campaign_summary.json')

    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
