# tests/test_mixing_bias.py
"""Coverage for MixedCalculator bias mode (E from base, F = F_base + lam*F_bias).

Uses lightweight stub calculators so the tests need neither torch, libfp,
nor mpi4py.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


_S_BASE = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
_S_BIAS = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0])


class _ConstCalc(Calculator):
    """Base stub: E=0, F=ones, stress=_S_BASE."""
    implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = {'energy': 0.0, 'forces': np.ones((n, 3)),
                        'stress': _S_BASE.copy()}


class _BiasCalc(Calculator):
    """Bias stub: F = 2*ones, stress=_S_BIAS."""
    implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = {'energy': 1.0, 'forces': 2.0 * np.ones((n, 3)),
                        'stress': _S_BIAS.copy()}


class _FailingCalc(Calculator):
    """Bias stub that always fails."""
    implemented_properties = ['energy', 'forces', 'stress']

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        raise RuntimeError("bias backend exploded")


def _atoms():
    return Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]],
                 cell=np.eye(3) * 5.0, pbc=True)


def test_bias_mode_raises_on_bias_failure():
    """A failing bias calculator must propagate, not silently degrade to a
    base-only (lambda=0) result that masquerades as a valid biased run."""
    from reformpy.mixing import MixedCalculator
    a = _atoms()
    a.calc = MixedCalculator(_ConstCalc(), _FailingCalc(), iter_max=10,
                             mode='bias', adaptive_lambda=True)
    with pytest.raises(RuntimeError, match="bias backend exploded"):
        a.get_forces()


def test_bias_mode_applies_lambda_when_healthy():
    """Forces = F_base + lambda*F_bias; energy is base-only."""
    from reformpy.mixing import MixedCalculator
    a = _atoms()
    mixed = MixedCalculator(_ConstCalc(), _BiasCalc(), iter_max=10,
                            mode='bias', adaptive_lambda=False)
    a.calc = mixed
    assert a.get_potential_energy() == 0.0         # base-only energy
    f = a.get_forces()
    lam = mixed.results['lambda']
    assert lam > 0.0
    assert np.allclose(f, 1.0 + lam * 2.0)         # F_base + lam*F_bias


def test_bias_mode_mixes_stress_when_enabled():
    """σ = σ_base + λ_σ·σ_bias so the fingerprint drives the cell."""
    from reformpy.mixing import MixedCalculator
    a = _atoms()
    mixed = MixedCalculator(_ConstCalc(), _BiasCalc(), iter_max=10,
                            mode='bias', adaptive_lambda=False, mix_stress=True)
    a.calc = mixed
    s = a.get_stress()
    lam_s = mixed.results['lambda_stress']
    assert lam_s > 0.0
    assert np.allclose(s, _S_BASE + lam_s * _S_BIAS)
    assert not np.allclose(s, _S_BASE)             # genuinely changed the cell stress


def test_bias_mode_stress_legacy_when_disabled():
    """mix_stress=False keeps the legacy base-only stress."""
    from reformpy.mixing import MixedCalculator
    a = _atoms()
    mixed = MixedCalculator(_ConstCalc(), _BiasCalc(), iter_max=10,
                            mode='bias', adaptive_lambda=False, mix_stress=False)
    a.calc = mixed
    s = a.get_stress()
    assert mixed.results['lambda_stress'] == 0.0
    assert np.allclose(s, _S_BASE)
