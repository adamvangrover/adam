import numpy as np

class SchrodingerSolver:
    """
    Simulates the time-evolution of a quantum search on a 2-level subspace.
    Used to calculate the success probability of a given annealing schedule.
    """

    def __init__(self, N):
        """
        Args:
            N (float): The size of the search space (e.g., 2**n).
        """
        self.N = N
        # Overlap between initial state |s> and target |w>
        # sin(theta) = 1/sqrt(N)
        self.sin_theta = 1.0 / np.sqrt(N)
        self.cos_theta = np.sqrt(1 - self.sin_theta**2)

    def hamiltonian(self, s_val):
        """
        Constructs the 2x2 Hamiltonian for a given schedule value s(t) in [0, 1].

        Basis: |w> (Target), |w_perp> (Orthogonal to target)
        H(s) = (1-s)H_0 + sH_P
        H_P = -|w><w| = [[-1, 0], [0, 0]] (Energy -1 for solution)
        H_0 = -|s><s|

        |s> = sin(theta)|w> + cos(theta)|w_perp>
        H_0 matrix elements:
        <w|H_0|w> = -sin^2(theta)
        <w_perp|H_0|w_perp> = -cos^2(theta)
        <w|H_0|w_perp> = -sin(theta)cos(theta)

        Total H(s):
        H_11 = -(1-s)*sin^2(theta) - s
        H_22 = -(1-s)*cos^2(theta)
        H_12 = H_21 = -(1-s)*sin(theta)*cos(theta)
        """
        h11 = -(1 - s_val) * (self.sin_theta**2) - s_val
        h22 = -(1 - s_val) * (self.cos_theta**2)
        h12 = -(1 - s_val) * (self.sin_theta * self.cos_theta)

        return np.array([[h11, h12], [h12, h22]])

    def evolve(self, schedule_params, time_steps=100, total_time=1.0):
        """
        Evolves the system from t=0 to t=T using the given schedule parameters.
        The schedule s(t) is parameterized.

        Args:
            schedule_params (np.ndarray): Coefficients for the schedule function.
                                          We'll use a simple polynomial or discretized schedule.
                                          For simplicity: discretized points linearly interpolated.
            time_steps (int): Number of Trotter steps.
            total_time (float): Total annealing time T.

        Returns:
            float: Probability of success |<w|psi(T)>|^2.
            float: Energy (Expectation value of H_P at final state).
        """
        dt = total_time / time_steps

        # Initial state: |s> = [sin(theta), cos(theta)]
        psi = np.array([self.sin_theta, self.cos_theta], dtype=np.complex128)

        # Interpolate schedule from params
        # Assume params are values of s(t) at equidistant points
        # If len(params) != time_steps, we interpolate
        t_points = np.linspace(0, 1, len(schedule_params))
        t_sim = np.linspace(0, 1, time_steps)
        s_schedule = np.interp(t_sim, t_points, schedule_params)

        # Enforce boundary conditions: s(0)=0, s(T)=1 is typical but Adam might optimize relaxation
        # For search, we enforce s(0)=0 and s(T)=1 usually, but let's let Adam optimize the middle.

        history = []

        for t_idx in range(time_steps):
            s_val = np.clip(s_schedule[t_idx], 0, 1)
            H = self.hamiltonian(s_val)

            # Unitary evolution U = exp(-i*H*dt)
            # Since H is small 2x2, we can compute expm explicitly or use approximation
            # U approx I - i*H*dt (Euler) -> unstable for long times
            # Better: Crank-Nicolson or simple eigen-decomposition for 2x2

            # Eigen decomposition for 2x2 is fast
            evals, evecs = np.linalg.eigh(H)
            U_diag = np.diag(np.exp(-1j * evals * dt))
            U = evecs @ U_diag @ evecs.conj().T

            psi = U @ psi

            # Track probability of being in |w> (index 0)
            prob_w = np.abs(psi[0])**2
            history.append(prob_w)

        final_prob = np.abs(psi[0])**2
        return final_prob, history, s_schedule

    def calculate_gradient(self, schedule_params, time_steps=50, total_time=1.0, epsilon=1e-4):
        """
        Calculates numerical gradient of the cost function (1 - P_success)
        with respect to schedule parameters.
        """
        base_prob, _, _ = self.evolve(schedule_params, time_steps, total_time)
        base_cost = 1.0 - base_prob

        grads = np.zeros_like(schedule_params)

        for i in range(len(schedule_params)):
            # Perturb parameter i
            params_perturbed = schedule_params.copy()
            params_perturbed[i] += epsilon

            prob_p, _, _ = self.evolve(params_perturbed, time_steps, total_time)
            cost_p = 1.0 - prob_p

            grads[i] = (cost_p - base_cost) / epsilon

        return grads, base_prob
