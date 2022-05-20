import numpy as np
import functools
from tabulate import tabulate

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


def round_pattern(min_val: float, max_val: float, num_of_segments: int, pattern: [float]) -> [float]:
    total_span = max_val - min_val
    segment_length = total_span / num_of_segments

    rounded_pattern = pattern.copy()
    for i in range(0, len(pattern)):
        for seg_len_multiple in range(1, num_of_segments + 1):
            category = seg_len_multiple * segment_length
            if pattern[i] <= category:
                rounded_pattern[i] = round(category, 2)
                break

    return rounded_pattern


min_val = 0.0
max_val = 10.0
seq_length = 10000
pattern_length = 100
seed = 123

sequence = random_sequence(min_val=min_val, max_val=max_val, seq_length=seq_length, seed=seed)
pattern = random_pattern(pattern_length=pattern_length, sequence=sequence, seed=seed)

num_of_segments = 3
rounded_pattern = round_pattern(min_val=min_val, max_val=max_val, num_of_segments=num_of_segments, pattern=pattern)


# ------------------------------------- smith waterman -------------------------------------
@functools.cache
def sub_matrix(a: float, b: float) -> int:
    return 1 if a == b else 0


@functools.cache
def gap_penalty(k: int) -> int:
    return 2 * k


def search_back_in_column(i: int, j: int) -> int:
    maximum = H[i - 1][j] - gap_penalty(1)
    for k in range(2, i + 1):
        new_val = H[i - k][j] - gap_penalty(k)
        if new_val > maximum:
            maximum = new_val
    return maximum


def search_back_in_row(i: int, j: int) -> int:
    maximum = H[i][j - 1] - gap_penalty(1)
    for k in range(2, i + 1):
        new_val = H[i][j - k] - gap_penalty(k)
        if new_val > maximum:
            maximum = new_val
    return maximum


H = np.zeros(shape=(len(pattern) + 1, len(sequence) + 1))

for i in range(1, len(pattern) + 1):
    for j in range(1, len(sequence) + 1):
        H[i][j] = max(
            H[i - 1][j - 1] + sub_matrix(sequence[j - 1], pattern[i - 1]),
            search_back_in_column(i, j),
            search_back_in_row(i, j),
            0
        )

# ------------------------------------- printing values ------------------------------------

rounded2_sequence = [round(x, 2) for x in sequence]  # for display purposes only
rounded2_pattern = [round(x, 2) for x in pattern]  # for display purposes only

print("Sequence: " + str(rounded2_sequence))
print("Pattern: " + str(rounded2_pattern))
print()
# print(tabulate(H, showindex=([""] + ronded2_pattern), headers=rounded2_sequence, tablefmt="presto"))
