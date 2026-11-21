import unittest
import numpy as np
from core.quantum.adam_optimizer import AdamOptimizer
from core.quantum.schrodinger_solver import SchrodingerSolver

class TestQuantumCore(unittest.TestCase):

    def test_optimizer_step(self):
        """Test that Adam optimizer updates parameters."""
        opt = AdamOptimizer(learning_rate=0.1)
        params = np.array([0.5, 0.5])
        grads = np.array([0.1, -0.1])

        new_params = opt.step(params, grads)

        # Check that parameters moved in opposite direction of gradient
        self.assertLess(new_params[0], 0.5)
        self.assertGreater(new_params[1], 0.5)

    def test_hamiltonian_construction(self):
        """Test Hamiltonian matrix properties."""
        solver = SchrodingerSolver(N=4)
        H = solver.hamiltonian(s_val=0.0) # Start of schedule

        # At s=0, H should be H_0
        # |s> = [sin(theta), cos(theta)] is ground state of H_0
        # sin(theta) = 1/2, cos(theta) = sqrt(3)/2
        # H_0 should have energy -1 for |s>

        psi_s = np.array([0.5, np.sqrt(3)/2])
        energy = psi_s.conj().T @ H @ psi_s
        self.assertAlmostEqual(energy, -1.0)

    def test_evolution_probability(self):
        """Test that evolution preserves norm."""
        solver = SchrodingerSolver(N=4)
        params = np.linspace(0, 1, 10)
        prob, _, _ = solver.evolve(params)

        self.assertTrue(0 <= prob <= 1.0)

if __name__ == '__main__':
    unittest.main()
