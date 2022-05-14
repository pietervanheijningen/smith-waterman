import numpy as np


# generates a random sequence of floats within a specified min, max and length
def random_sequence(min_val: float, max_val: float, seq_length: int, seed: int) -> [float]:
    np.random.seed(seed)
    return list(map(
        lambda x: x * (max_val - min_val) + min_val,
        np.random.random_sample(seq_length)
    ))


def random_pattern(pattern_length: int, sequence: [float], seed: int) -> [float]:
    np.random.seed(seed)
    index = np.random.randint(pattern_length - 1, len(sequence))
    return sequence[(index - pattern_length + 1):index]


print(random_pattern(pattern_length=10, sequence=random_sequence(min_val=0.0, max_val=10.0, seq_length=100, seed=123), seed=123))
