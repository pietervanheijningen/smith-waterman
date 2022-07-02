import numpy as np
import functools
import time
import csv
import sys

def random_sequence(seq_length: int) -> [float]:
    return list(map(
        lambda x: x * (max_val - min_val) + min_val,
        np.random.random_sample(seq_length)
    ))


def random_pattern(pattern_length: int, sequence: [float]) -> [float]:
    index = np.random.randint(pattern_length - 1, len(sequence))
    return sequence[(index - pattern_length + 1):index + 1], index


def add_modifications(float_array: [float]) -> [float]:
    # workaround for if the seed is fixed.
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


@functools.cache
def sub_matrix(a: float, b: float) -> int:
    return 2 - int((abs(a - b) / segment_length))


@functools.cache
def gap_penalty(k: int) -> int:
    u = 2
    v = 0

    return (u * k) + v


def search_back_in_column(i: int, j: int) -> int:
    maximum = H[i - 1][j] - gap_penalty(1)
    return maximum  # comment this out to make it look back further
    for k in range(2, i + 1):
        new_val = H[i - k][j] - gap_penalty(k)
        if new_val > maximum:
            maximum = new_val
    return maximum


def search_back_in_row(i: int, j: int) -> int:
    maximum = H[i][j - 1] - gap_penalty(1)
    return maximum  # comment this out to make it look back further
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
    bottom_row = H[len(rounded_pattern)]
    return list(map(lambda a: a[0] - 1, np.argwhere(bottom_row == np.amax(bottom_row)))), \
           np.max(bottom_row)


# ------------------------------------- adjustable variables -------------------------------
fh = open('real-world-data/ref_stream.data', 'rb')
pattern = list(map(lambda x: float(x), bytearray(fh.read())))
fh.close()

fh2 = open('real-world-data/adc_stream.data', 'rb')
original_sequence = list(map(lambda x: float(x), bytearray(fh2.read())))
fh2.close()


min_val = min(original_sequence)
max_val = max(original_sequence)
seq_length = len(original_sequence)
pattern_length = len(pattern)
num_of_iterations = 10
num_of_segments = 16
prob_modification = 0.0
prob_repetition = 0.0
prob_further_repetitions = 0
margin_of_error = 0.01

print(seq_length)
print(pattern_length)

# ------------------------------------- /adjustable variables ------------------------------
margin_of_error_int = int(margin_of_error * seq_length)
segment_length = (max_val - min_val) / num_of_segments

original_actual_indexes = [541, 1091, 1641, 2191, 2741]

if len(sys.argv) > 1:
    print("Using set seed: " + sys.argv[1])
    np.random.seed(int(sys.argv[1]))
else:
    seed = np.random.randint(1, 2 ** 32)
    print("Using randomly generated seed: " + str(seed))
    np.random.seed(seed)

with open("results/" + str(int(time.time())) + '.csv', 'w') as file:
    writer = csv.writer(file)

    for j in range(0, 21):

        prob_modification = 0.05 * j
        prob_repetition = 0.05 * j

        print("prob: " + str(0.05 * j))

        hit_rates = []
        for i in range(1, num_of_iterations + 1):
            sequence, to_old_index_map = add_repetitions(add_modifications(original_sequence))

            H = np.zeros(shape=(len(pattern) + 1, len(sequence) + 1))

            rounded_pattern = round_up(num_of_segments=num_of_segments, float_array=pattern)
            rounded_sequence = round_up(num_of_segments=num_of_segments, float_array=sequence)

            match_indexes, max_val_matrix = do_smith_waterman(rounded_sequence, rounded_pattern)

            actual_indexes = list(map(lambda x: to_old_index_map.index(x), original_actual_indexes))

            hits = 0
            for match_index in match_indexes:
                for actual_index in actual_indexes:
                    if (actual_index - margin_of_error_int) <= match_index <= (actual_index + margin_of_error_int):
                        hits += 1

            hit_rates.append(hits)

            print("Matched indexes: " + str(match_indexes))
            print("hits: " + str(hits))
            print()

        print(np.average(hit_rates))
        writer.writerow([round(prob_modification, 2), round(np.average(hit_rates), 2)])
