import time
import random
import numpy as np

def benchmark_scalar():
    mu = 0.0
    sigma = 1.0
    iterations = 1_000_000

    start = time.time()
    for _ in range(iterations):
        random.gauss(mu, sigma)
    end = time.time()
    print(f"random.gauss: {end - start:.4f}s")

    start = time.time()
    for _ in range(iterations):
        np.random.normal(mu, sigma)
    end = time.time()
    print(f"np.random.normal: {end - start:.4f}s")

if __name__ == "__main__":
    benchmark_scalar()
