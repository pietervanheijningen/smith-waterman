import numpy as np
import functools
import csv
import time
import sys

from tabulate import tabulate


def random_sequence(seq_length: int) -> [float]:
    return list(map(
        lambda x: x * (max_val - min_val) + min_val,
        np.random.random_sample(seq_length)
    ))


def random_pattern(pattern_length: int, sequence: [float]) -> [float]:
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
            while prob_further_repetitions >= np.random.random_sample():
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
seq_length = 20
pattern_length = 5
# seed = 123
num_of_segments = 10
prob_modification = 0.2
prob_repetition = 0.2
prob_further_repetitions = 0.1
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
    return list(map(lambda a: a[1], np.argwhere(H == np.amax(H[len(rounded_pattern)])))), \
           np.max(H[len(rounded_pattern)])


# ------------------------------------- printing values ------------------------------------

with open("results/" + str(int(time.time())) + '.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["iteration", "actual_index", "distance_between_avg_matched_indexes"])

    if len(sys.argv) > 1:
        print("Using set seed: " + sys.argv[1])
        np.random.seed(int(sys.argv[1]))
    else:
        seed = np.random.randint(1, 2 ** 32)
        print("Using randomly generated seed: " + str(seed))
        np.random.seed(seed)

    for i in range(0, 4):
        print("Iteration #" + str(i) + "..")
        sequence = random_sequence(seq_length=seq_length)
        pattern, pattern_index = random_pattern(pattern_length=pattern_length, sequence=sequence)

        sequence = add_modifications(sequence)
        sequence, to_old_index_map = add_repetitions(sequence)

        H = np.zeros(shape=(len(pattern) + 1, len(sequence) + 1))

        rounded_pattern = round_up(num_of_segments=num_of_segments, float_array=pattern)
        rounded_sequence = round_up(num_of_segments=num_of_segments, float_array=sequence)

        match_indexes, max_val_matrix = do_smith_waterman(rounded_sequence, rounded_pattern)
        confidence_smith_waterman = round((max_val_matrix / (2 * (pattern_length-1))) * 100, 2)

        actual_index = to_old_index_map.index(pattern_index)

        closest_match = min(match_indexes, key=lambda x: abs(x - actual_index))
        furthest_match = max(match_indexes, key=lambda x: abs(x - actual_index))
        # print("Sequence: " + str(rounded_sequence))
        # print("Pattern: " + str(rounded_pattern))
        print("Matched indexes: " + str(match_indexes))
        print("Actual index: " + str(actual_index))
        print("Closest match: " + str(closest_match))
        print("Furthest match: " + str(furthest_match))
        print("Smith waterman confidence: " + str(confidence_smith_waterman) + "%")
        print(tabulate(H, showindex=([""] + rounded_pattern), headers=rounded_sequence, tablefmt="presto"))
        print()
        writer.writerow([i, actual_index, match_indexes])
