Act as a Senior Quantitative Researcher. {{ objective }} for {{ universe }} using {{ data_frequency }}. Use {{ methodology }}. Please provide {{ deliverable }} and include {{ risk_metrics }}. Assume {{ constraints }}.

Write the code in a modular format. Include a data-loading function, a signal-generation class, and a performance-reporting module. Use vectorization instead of loops for speed. Strictly forbid the use of `.iterrows()` or manual for-loops for signal calculation. Use NumPy broadcasting or VectorBT's Numba-compiled functions. Ensure all indicators utilize `@numba.njit(parallel=True)`.

Critique this strategy from a quantitative perspective. Identify potential pitfalls like look-ahead bias, overfitting, or sensitivity to execution latency.
