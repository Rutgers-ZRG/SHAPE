import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError
from ase.parallel import world

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    # Serial fallback: the calculators below fully support comm=None
    # (rank 0, no barriers); mpi4py is only required for parallel runs.
    MPI = None
    comm = None
    rank = 0


class LinearCombinationCalculator(Calculator):
    """LinearCombinationCalculator for weighted summation of multiple calculators.
    """

    def __init__(self, calcs, weights, atoms = None):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        weights: list of float
            Weights for each calculator in the list.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        super().__init__(atoms = atoms)

        if len(calcs) == 0:
            raise ValueError('The value of the calcs must be a list of Calculators')

        for calc in calcs:
            if not isinstance(calc, Calculator):
                raise ValueError('All the calculators should be inherited form the \
                                  ase\'s Calculator class')

        common_properties = set.intersection(*(set(calc.implemented_properties) for calc in calcs))
        self.implemented_properties = list(common_properties)

        if not self.implemented_properties:
            raise PropertyNotImplementedError('There are no common property \
                                               implemented for the potentials!')
        if len(weights) != len(calcs):
            raise ValueError('The length of the weights must be the same as \
                              the number of calculators!')

        self.calcs = calcs
        self.weights = weights

        '''
        weights = np.ones(len(calcs)).tolist()
        pi_fmax = 1.0
        for i in range(len(calcs)):
            forces_i = calcs[i].get_property('forces', atoms)
            pi_fmax = pi_fmax*np.amax(np.absolute(forces_i))
        for j in range(len(weights)):
            forces_j = calcs[j].get_property('forces', atoms)
            weights[j] = pi_fmax / np.amax(np.absolute(forces_j))

        if len(weights) != len(calcs):
            raise ValueError('The length of the weights must be the same as \
                              the number of calculators!')

        # self.calcs = calcs
        self.weights = weights
        '''

    def calculate(self,
                  atoms = None,
                  properties = ['energy', 'forces', 'stress'],
                  system_changes = all_changes
                 ):
        """ Calculates all the specific property for each calculator
            and returns with the summed value.
        """

        super().calculate(atoms, properties, system_changes)

        if not set(properties).issubset(self.implemented_properties):
            raise PropertyNotImplementedError('Some of the requested property is not \
                                               in the ''list of supported properties \
                                               ({})'.format(self.implemented_properties))

        for w, calc in zip(self.weights, self.calcs):
            if w > 0.0:
                if calc.calculation_required(atoms, properties):
                    calc.calculate(atoms, properties, system_changes)

                for k in properties:
                    if k not in self.results:
                        self.results[k] = w * calc.results[k]
                    else:
                        self.results[k] += w * calc.results[k]

    def reset(self):
        """Clear all previous results recursively from all fo the calculators."""
        super().reset()

        for calc in self.calcs:
            calc.reset()

    def __str__(self):
        calculators = ', '.join(calc.__class__.__name__ for calc in self.calcs)
        return '{}({})'.format(self.__class__.__name__, calculators)


class MixedCalculator(LinearCombinationCalculator):
    """
    Mixing of two calculators with different weights

    H = weight1 * H1 + weight2 * H2

    Has functionality to get the energy contributions from each calculator

    Parameters
    ----------
    calc1 : ASE-calculator
    calc2 : ASE-calculator
    iter_max : int, optional
        Maximum number of iterations for weight transition (default: 300)
    comm : MPI communicator, optional
        MPI communicator for parallel calculations
    scheme : str, optional
        Weight mixing scheme. Options:
        - 'linear': w1 = t (simple linear ramp)
        - 'cosine': w1 = 0.5*(1 - cos(π*t)) (smooth, recommended)
        - 'smoothstep': w1 = 3t² - 2t³ (Hermite interpolation)
        - 'smootherstep': w1 = 6t⁵ - 15t⁴ + 10t³ (smoother transition)
        - 'sine_oscillate': original oscillatory scheme (legacy)
        Default is 'cosine'.
    """

    VALID_SCHEMES = ('linear', 'cosine', 'smoothstep', 'smootherstep', 'sine_oscillate')
    VALID_MODES = ('transition', 'bias')

    def __init__(self, calc1, calc2, iter_max=None, comm=None, scheme='cosine',
                 mode='transition', adaptive_lambda=False, eta=0.3,
                 max_f_bias_rms=50.0, mix_stress=True, max_s_bias_rms=5.0):
        """
        Parameters
        ----------
        calc1, calc2 : ASE Calculator
            In 'transition' mode: calc1 is target, calc2 is initial.
            In 'bias' mode: calc1 is base (physical), calc2 is bias.
        iter_max : int or None
            Number of steps for weight schedule.
        comm : MPI communicator or None
        scheme : str
            Weight schedule: 'linear', 'cosine', 'smoothstep', etc.
        mode : str
            'transition': w1 + w2 = 1, ramp from calc2 to calc1 (default).
            'bias': E from calc1 only, F = F1 + λ(t)·F2, λ anneals to 0.
        adaptive_lambda : bool
            If True (bias mode only), scale λ by force-RMS ratio:
            λ = η · |F1|_rms / |F2|_rms · schedule(t)
        eta : float
            Force-scale ratio for adaptive λ (default 0.3).
        max_f_bias_rms : float
            Safety clamp for bias forces (eV/Å, default 50.0).
        mix_stress : bool
            If True (bias mode), the cell is driven by σ = σ1 + λ_σ·σ2 so the
            fingerprint stress influences cell relaxation (FrechetCellFilter),
            not just atomic positions. λ_σ uses the same schedule and (when
            adaptive_lambda) the same η, scaled to the base-stress magnitude.
            Set False to keep the legacy behavior (stress from calc1 only).
        max_s_bias_rms : float
            Safety clamp for bias stress (eV/Å³, default 5.0 ≈ 800 GPa).
        """
        self.iter = 0
        self._last_positions = None
        if iter_max is not None:
            if iter_max <= 0:
                raise ValueError('iter_max must be a positive integer')
            self.iter_max = int(iter_max)
        else:
            self.iter_max = 300

        # Validate and store mixing scheme
        if scheme not in self.VALID_SCHEMES:
            raise ValueError(f"scheme must be one of {self.VALID_SCHEMES}, got '{scheme}'")
        self.scheme = scheme

        # Validate and store mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.mode = mode
        self.adaptive_lambda = adaptive_lambda
        self.eta = eta
        self.mix_stress = mix_stress
        self.max_s_bias_rms = max_s_bias_rms
        self.max_f_bias_rms = max_f_bias_rms

        # Store MPI communicator if provided
        self.comm = comm
        if comm is not None:
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
            self.parallel = True
        else:
            self.rank = 0
            self.size = 1
            self.parallel = False

        self.weights = [0.0, 1.0]
        self._last_weights = tuple(self.weights)
        self._weights_changed_last_call = True
        weight1 = self.weights[0]
        weight2 = self.weights[1]
        super().__init__([calc1, calc2], [weight1, weight2])

    def _compute_weight(self, progress):
        """Compute weight for calc1 based on progress (0 to 1) and scheme."""
        t = progress
        if self.scheme == 'linear':
            return t
        elif self.scheme == 'cosine':
            # Smooth transition with zero derivative at endpoints
            return 0.5 * (1.0 - np.cos(np.pi * t))
        elif self.scheme == 'smoothstep':
            # Hermite interpolation: 3t² - 2t³
            return t * t * (3.0 - 2.0 * t)
        elif self.scheme == 'smootherstep':
            # Ken Perlin's improved smoothstep: 6t⁵ - 15t⁴ + 10t³
            return t * t * t * (t * (6.0 * t - 15.0) + 10.0)
        elif self.scheme == 'sine_oscillate':
            # Original oscillatory scheme (legacy)
            return 0.5 * (np.sin(-np.pi * 0.5 + np.pi * 9 * t ** 2) + 1.0)
        else:
            # Fallback to cosine
            return 0.5 * (1.0 - np.cos(np.pi * t))

    def set_weights(self, calc1, calc2, atoms):
        """Set weights for the two calculators based on iteration number."""
        effective_iter = min(self.iter, self.iter_max)

        if effective_iter == 0:
            weight0 = 0.0
        elif effective_iter >= self.iter_max:
            weight0 = 1.0
        else:
            # Compute progress from 0 to 1
            progress = effective_iter / max(self.iter_max - 1, 1)
            weight0 = self._compute_weight(progress)

        # Ensure numerical safety and that the two weights sum to 1.0
        weight0 = float(np.clip(weight0, 0.0, 1.0))
        weight1 = 1.0 - weight0

        prev_weights = self._last_weights
        new_weights = (weight0, weight1)

        self.weights[0] = weight0
        self.weights[1] = weight1
        self._last_weights = new_weights

        changed = not np.allclose(prev_weights, new_weights, rtol=0.0, atol=1e-12)
        self._weights_changed_last_call = changed

        # Ensure weights are synchronized if using MPI
        if self.parallel:
            self.weights = self.comm.bcast(self.weights if self.rank == 0 else None, root=0)
            changed = self.comm.bcast(changed if self.rank == 0 else None, root=0)
            self._weights_changed_last_call = bool(changed)

    def _update_iteration_counter(self, atoms):
        """Increment iteration counter only when atomic positions change."""
        positions = atoms.get_positions()

        if self._last_positions is None:
            changed = False
        else:
            changed = not np.allclose(positions, self._last_positions,
                                       rtol=0.0, atol=1e-12)

        if self.parallel:
            changed = self.comm.bcast(changed if self.rank == 0 else None, root=0)

        if self._last_positions is None or changed:
            self._last_positions = np.array(positions, copy=True)

        if changed:
            self.iter += 1

        # Synchronize iteration counter across ranks
        if self.parallel:
            self.iter = self.comm.bcast(self.iter if self.rank == 0 else None, root=0)
                
    def calculate(self,
                  atoms = None,
                  properties = ['energy', 'forces', 'stress'],
                  system_changes = all_changes
                 ):
        """ Calculates all the specific property for each calculator and returns
            with the summed value.

        In 'transition' mode (default):
            E = w1*E1 + w2*E2,  F = w1*F1 + w2*F2  (w1+w2=1)

        In 'bias' mode:
            E = E1 (base only),  F = F1 + λ·F2,  λ anneals to 0
            If adaptive_lambda: λ = η · |F1|_rms / |F2|_rms · schedule(t)
            Else: λ = schedule(t)  (schedule goes from 1 → 0)
        """
        atoms = atoms or self.atoms
        if atoms is None:
            raise ValueError("No atoms object provided to MixedCalculator")

        # Make sure all ranks have the same atoms
        if self.parallel:
            # Wait for all processes to reach this point
            self.comm.Barrier()

        # Update iteration counter before adjusting weights
        self._update_iteration_counter(atoms)

        # Set weights based on current iteration
        self.set_weights(self.calcs[0], self.calcs[1], atoms)

        # Initialize dictionaries on all ranks
        self.results = {}

        if self.mode == 'bias':
            self._calculate_bias(atoms, properties, system_changes)
        else:
            self._calculate_transition(atoms, properties, system_changes)

        # Ensure all processes have completed the calculation
        if self.parallel:
            self.comm.Barrier()

    def _calculate_transition(self, atoms, properties, system_changes):
        """Original transition mode: weighted sum of both calculators."""
        # Calculate properties on all ranks for both calculators
        for i, (w, calc) in enumerate(zip(self.weights, self.calcs)):
            if w > 0.0:
                if calc.calculation_required(atoms, properties):
                    try:
                        calc.calculate(atoms, properties, system_changes)
                    except Exception as e:
                        if self.rank == 0:
                            print(f"Error in calculator {i+1}: {e}", flush=True)
                        if self.parallel:
                            self.comm.Barrier()
                        raise
                if self.parallel:
                    self.comm.Barrier()

        # Collect and combine results
        for prop in properties:
            if prop in ['energy', 'forces', 'stress']:
                if prop == 'energy':
                    result = 0.0
                elif prop == 'forces':
                    result = np.zeros((len(atoms), 3), dtype=float)
                elif prop == 'stress':
                    result = np.zeros(6, dtype=float)

                for i, (w, calc) in enumerate(zip(self.weights, self.calcs)):
                    if w > 0.0:
                        if prop in calc.results:
                            result += w * calc.results[prop]

                self.results[prop] = result

                if prop == 'energy':
                    energy_contributions = tuple(
                        calc.results.get('energy', 0.0) if w > 0.0 else 0.0
                        for w, calc in zip(self.weights, self.calcs)
                    )
                    self.results['energy_contributions'] = energy_contributions
                elif prop == 'forces':
                    force_contributions = tuple(
                        calc.results.get('forces', np.zeros((len(atoms), 3), dtype=float))
                        if w > 0.0 else np.zeros((len(atoms), 3), dtype=float)
                        for w, calc in zip(self.weights, self.calcs)
                    )
                    self.results['force_contributions'] = force_contributions
                elif prop == 'stress':
                    stress_contributions = tuple(
                        calc.results.get('stress', np.zeros(6, dtype=float))
                        if w > 0.0 else np.zeros(6, dtype=float)
                        for w, calc in zip(self.weights, self.calcs)
                    )
                    self.results['stress_contributions'] = stress_contributions

    def _calculate_bias(self, atoms, properties, system_changes):
        """Bias mode: E from calc1, F = F1 + λ·F2, λ anneals to 0."""
        base_calc = self.calcs[0]  # physical PES
        bias_calc = self.calcs[1]  # bias (Reform/entropy)

        # Always run base calculator
        if base_calc.calculation_required(atoms, properties):
            try:
                base_calc.calculate(atoms, properties, system_changes)
            except Exception as e:
                if self.rank == 0:
                    print(f"Error in base calculator: {e}", flush=True)
                raise
        if self.parallel:
            self.comm.Barrier()

        # Compute bias schedule: how much bias remains (1 → 0)
        effective_iter = min(self.iter, self.iter_max)
        if self.iter_max > 0:
            progress = effective_iter / max(self.iter_max, 1)
            # Invert the schedule: we want bias_weight to go from 1 → 0
            bias_schedule = 1.0 - self._compute_weight(progress)
        else:
            bias_schedule = 1.0

        # Run bias calculator if schedule > 0 and forces/stress needed
        need_bias = bias_schedule > 1e-10 and (
            'forces' in properties or 'stress' in properties)

        if need_bias:
            if bias_calc.calculation_required(atoms, properties):
                try:
                    bias_calc.calculate(atoms, properties, system_changes)
                except Exception as e:
                    # Fail loud: a bias-calculator error must NOT be silently
                    # swallowed into a base-only (lambda=0) result, which would
                    # masquerade as a valid biased run and corrupt any
                    # bias-vs-control comparison. Matches the base-calculator
                    # path, which also re-raises.
                    if self.rank == 0:
                        print(f"Error in bias calculator: {e}", flush=True)
                    raise
            if self.parallel:
                self.comm.Barrier()

        # Energy: always from base only
        if 'energy' in properties:
            self.results['energy'] = base_calc.results.get('energy', 0.0)
            self.results['energy_contributions'] = (
                base_calc.results.get('energy', 0.0),
                bias_calc.results.get('energy', 0.0) if need_bias else 0.0
            )

        # Forces: F_base + λ · F_bias
        if 'forces' in properties:
            f_base = base_calc.results.get(
                'forces', np.zeros((len(atoms), 3), dtype=float))
            if need_bias and 'forces' in bias_calc.results:
                f_bias = bias_calc.results['forces'].copy()

                # Safety clamp
                f_bias_rms = np.sqrt(np.mean(f_bias ** 2)) + 1e-30
                if f_bias_rms > self.max_f_bias_rms:
                    f_bias *= self.max_f_bias_rms / f_bias_rms
                    f_bias_rms = self.max_f_bias_rms

                # Compute λ
                if self.adaptive_lambda:
                    f_base_rms = np.sqrt(np.mean(f_base ** 2)) + 1e-30
                    lam = self.eta * f_base_rms / f_bias_rms * bias_schedule
                else:
                    lam = bias_schedule

                self.results['forces'] = f_base + lam * f_bias
                self.results['lambda'] = lam
            else:
                self.results['forces'] = f_base
                self.results['lambda'] = 0.0

            self.results['force_contributions'] = (
                f_base,
                bias_calc.results.get('forces', np.zeros((len(atoms), 3), dtype=float))
                if need_bias else np.zeros((len(atoms), 3), dtype=float)
            )

        # Stress: σ_base + λ_σ · σ_bias when mix_stress (lets the fingerprint
        # drive the cell during the biased phase, mirroring the force bias;
        # λ_σ uses the same schedule, scaled adaptively to the base-stress
        # magnitude, and anneals to 0). Legacy behavior (stress from base
        # only) when mix_stress is False.
        if 'stress' in properties:
            s_base = base_calc.results.get('stress', np.zeros(6, dtype=float))
            if self.mix_stress and need_bias and 'stress' in bias_calc.results:
                s_bias = bias_calc.results['stress'].copy()

                # Safety clamp
                s_bias_rms = np.sqrt(np.mean(s_bias ** 2)) + 1e-30
                if s_bias_rms > self.max_s_bias_rms:
                    s_bias *= self.max_s_bias_rms / s_bias_rms
                    s_bias_rms = self.max_s_bias_rms

                # Compute λ_σ
                if self.adaptive_lambda:
                    s_base_rms = np.sqrt(np.mean(s_base ** 2)) + 1e-30
                    lam_s = self.eta * s_base_rms / s_bias_rms * bias_schedule
                else:
                    lam_s = bias_schedule

                self.results['stress'] = s_base + lam_s * s_bias
                self.results['lambda_stress'] = lam_s
            else:
                self.results['stress'] = s_base
                self.results['lambda_stress'] = 0.0

            self.results['stress_contributions'] = (
                s_base,
                bias_calc.results.get('stress', np.zeros(6, dtype=float))
                if need_bias else np.zeros(6, dtype=float)
            )

    def reset(self):
        """Reset calculators and internal mixing state."""
        super().reset()
        self.iter = 0
        self._last_positions = None
        self.weights = [0.0, 1.0]
        self._last_weights = tuple(self.weights)
        self._weights_changed_last_call = True

    def get_energy_contributions(self, atoms = None):
        """ Return the potential energy from calc1 and calc2 respectively """
        self.calculate(properties = ['energy'], atoms = atoms)
        return self.results['energy_contributions']

    def get_force_contributions(self, atoms = None):
        """ Return the forces from calc1 and calc2 respectively """
        self.calculate(properties = ['forces'], atoms = atoms)
        return self.results['force_contributions']

    def get_stress_contributions(self, atoms = None):
        """ Return the Cauchy stress tensor from calc1 and calc2 respectively """
        self.calculate(properties = ['stress'], atoms = atoms)
        return self.results['stress_contributions']



class SumCalculator(LinearCombinationCalculator):
    """SumCalculator for combining multiple calculators.

    This calculator can be used when there are different calculators for the different chemical
    environment or for example during delta leaning. It works with a list of arbitrary calculators
    and evaluates them in sequence when it is required. The supported properties are the intersection
    of the implemented properties in each calculator.
    """

    def __init__(self, calcs, atoms = None):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        weights = [1.] * len(calcs)
        super().__init__(calcs, weights, atoms)


class AverageCalculator(LinearCombinationCalculator):
    """AverageCalculator for equal summation of multiple calculators (for thermodynamic purposes)..
    """

    def __init__(self, calcs, atoms = None):
        """Implementation of average of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        atoms: Atoms object
            Optional :class:`~ase.Atoms` object to which the calculator will be attached.
        """

        n = len(calcs)

        if n == 0:
            raise ValueError('The value of the calcs must be a list of Calculators')

        weights = [1 / n] * n
        super().__init__(calcs, weights, atoms)
