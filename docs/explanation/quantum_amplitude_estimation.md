# Quantum Amplitude Estimation (QAE) in ADAM

## Overview
ADAM incorporates experimental modules (`adam-quantum`) designed to explore the frontiers of tail-risk pricing using quantum computing paradigms, specifically Quantum Amplitude Estimation (QAE).

## Mathematical Theory
Classical Monte Carlo simulations are heavily utilized for pricing complex derivatives and estimating Value-at-Risk (VaR) in distressed debt portfolios. However, Monte Carlo converges at a rate of $O(1/\sqrt{N})$, where $N$ is the number of samples.

Quantum Amplitude Estimation offers a theoretical quadratic speedup, achieving convergence at $O(1/N)$. By encoding the financial probability distribution into a quantum state and applying Grover-like operators, QAE can estimate the expectation value of the payout function significantly faster.

In ADAM, we integrate with Qiskit and cuQuantum to construct Hamiltonian-based optimization models. We map the credit default probabilities (derived deterministically from our System 2 graphs) into quantum circuits to simulate extreme market tail-risks and correlated default scenarios across Broadly Syndicated Loans (BSL).

## Current Limitations
While theoretically promising, the integration is highly experimental:
1. **NISQ Era Constraints:** Current Noisy Intermediate-Scale Quantum (NISQ) devices suffer from high gate error rates and decoherence, making deep circuits for complex BSL portfolios impractical on physical hardware.
2. **State Preparation:** Loading classical financial data (e.g., a massive covariance matrix) into a quantum state efficiently remains a significant bottleneck.
3. **Resource Costs:** Simulating these circuits locally via cuQuantum is computationally expensive and requires significant GPU resources.

As quantum hardware matures, the `adam-quantum` module is positioned to transition from an experimental tail-risk simulator to a production-ready pricing engine.
