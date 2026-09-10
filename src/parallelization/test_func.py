import math

def compute_primes(limit):
    """Sieve of Eratosthenes – memory‑intensive + CPU‑heavy."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit+1:step] = b'\x00' * ((limit - start) // step + 1)
    return sum(sieve)   # count of primes

def heavy_flops():
    """Floating‑point heavy loop – billions of operations."""
    total = 0.0
    for i in range(1, 30_000_000):
        total += math.sqrt(i) * math.sin(i)
    return total

if __name__ == "__main__":
    prime_count = compute_primes(2_000_000)   # primes up to 2 million
    flops_result = heavy_flops()              # 30 million sqrt+sin
    print(f"Primes found: {prime_count}")
    print(f"Floating sum: {flops_result:.2f}")