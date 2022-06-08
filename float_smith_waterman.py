import numpy as np
import functools
import csv
import time
from tabulate import tabulate


def random_sequence(seq_length: int) -> [float]:
    # np.random.seed(seed)
    return list(map(
        lambda x: x * (max_val - min_val) + min_val,
        np.random.random_sample(seq_length)
    ))


def random_pattern(pattern_length: int, sequence: [float]) -> [float]:
    # np.random.seed(seed)
    index = np.random.randint(pattern_length - 1, len(sequence))
    return sequence[(index - pattern_length + 1):index], index


def add_modifications(float_array: [float]) -> [float]:
    # workaround for if the float is fixed.
    rand_array = random_sequence(len(float_array) * 2)[len(float_array):]

    modified_array = np.zeros(len(float_array))
    for i in range(0, len(float_array)):
        if prob_modification >= np.random.random_sample():
            modified_array[i] = rand_array[i]
        else:
            modified_array[i] = float_array[i]
    return modified_array


def add_repetitions(float_array: [float]) -> [float]:
    modified_array = []
    to_old_index_map = []
    j = 0
    for i in range(0, len(float_array)):
        modified_array.append(float_array[i])
        to_old_index_map.append(j)
        if prob_repetition >= np.random.random_sample():
            modified_array.append(float_array[i])
            to_old_index_map.append(-1)
        j += 1
    return modified_array, to_old_index_map


def round_up(num_of_segments: int, float_array: [float]) -> [float]:
    rounded_array = float_array.copy()
    for i in range(0, len(float_array)):
        for seg_len_multiple in range(1, num_of_segments + 1):
            category = seg_len_multiple * segment_length
            if float_array[i] <= category:
                rounded_array[i] = round(category, 2)
                break

    return rounded_array


# ------------------------------------- adjustable variables -------------------------------

min_val = 0.0
max_val = 100.0
seq_length = 100
pattern_length = 20
# seed = 123
num_of_segments = 10
prob_modification = 0.6
prob_repetition = 0.6
segment_length = (max_val - min_val) / num_of_segments

# ------------------------------------- /adjustable variables ------------------------------
# ------------------------------------- smith waterman -------------------------------------



@functools.cache
def sub_matrix(a: float, b: float) -> int:
    return 2 - int((abs(a - b) / segment_length))


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


def do_smith_waterman(rounded_sequence: [float], rounded_pattern: [float]):

    for i in range(1, len(pattern) + 1):
        for j in range(1, len(sequence) + 1):
            H[i][j] = max(
                H[i - 1][j - 1] + sub_matrix(rounded_sequence[j - 1], rounded_pattern[i - 1]),
                search_back_in_column(i, j),
                search_back_in_row(i, j),
                0
            )
    print(np.max(H[len(rounded_pattern)]))
    return list(map(lambda a: a[1], np.argwhere(H == np.amax(H[len(rounded_pattern)]))))


# ------------------------------------- printing values ------------------------------------

with open("results/" + str(int(time.time())) + '.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["iteration", "actual_index", "distance_between_avg_matched_indexes"])
    for i in range(0, 1):
        print(i)
        sequence = random_sequence(seq_length=seq_length)
        pattern, pattern_index = random_pattern(pattern_length=pattern_length, sequence=sequence)

        sequence = add_modifications(sequence)
        sequence, to_old_index_map = add_repetitions(sequence)

        print(to_old_index_map)
        print("random index: " + str(pattern_index))

        H = np.zeros(shape=(len(pattern) + 1, len(sequence) + 1))

        rounded_pattern = round_up(num_of_segments=num_of_segments, float_array=pattern)
        rounded_sequence = round_up(num_of_segments=num_of_segments, float_array=sequence)

        match_indexes = do_smith_waterman(rounded_sequence, rounded_pattern)

        actual_index = to_old_index_map.index(pattern_index)
        # print(tabulate(H, showindex=([""] + rounded_pattern), headers=rounded_sequence, tablefmt="presto"))

        writer.writerow([i, actual_index, match_indexes])
# print("(actual) Sequence: " + str(rounded2_sequence))
# print("(actual) Pattern: " + str(rounded2_pattern))
# print("Sequence: " + str(rounded_sequence))
# print("Pattern: " + str(rounded_pattern))
# print()
# print("Matched indexes: " + str(match_indexes))
# print("Actual index: " + str(to_old_index_map.index(pattern_index)))

# print(tabulate(H, showindex=([""] + rounded_pattern), headers=rounded_sequence, tablefmt="presto"))
