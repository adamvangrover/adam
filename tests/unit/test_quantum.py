import pytest
import numpy as np
from core.quantum.schrodinger_solver import SchrodingerSolver

@pytest.fixture
def solver():
    return SchrodingerSolver(N=4)

def test_initialization(solver):
    assert solver.N == 4
    assert np.isclose(solver.sin_theta, 0.5)

def test_hamiltonian(solver):
    # s=0 -> H_0
    H0 = solver.hamiltonian(0)
    assert H0.shape == (2, 2)
    # H_0 should have -1 eigenvalues? Let's just check shape and type for now
    assert isinstance(H0, np.ndarray)

    # s=1 -> H_P
    HP = solver.hamiltonian(1)
    # H_P = -|w><w| = [[-1, 0], [0, 0]]
    expected_HP = np.array([[-1, 0], [0, 0]])
    # Depending on implementation details, check close
    assert np.allclose(HP, expected_HP, atol=1e-5)

def test_evolution(solver):
    schedule = np.linspace(0, 1, 10)
    prob, history, s_schedule = solver.evolve(schedule, time_steps=20, total_time=1.0)

    assert 0 <= prob <= 1.0
    assert len(history) == 20
    assert len(s_schedule) == 20

def test_gradient(solver):
    schedule = np.linspace(0, 1, 5)
    grads, base_prob = solver.calculate_gradient(schedule, time_steps=10)

    assert len(grads) == 5
    assert isinstance(base_prob, float)
