import numpy as np

string = "CBDBAC"
pattern = "BCB"


# substitution matrix that is only supposed to work with A,B,C,D
def sub_matrix(a, b):
    # if ord(c)
    value = abs(ord(a) - ord(b))
    if value == 0:
        return 2
    if value == 1:
        return 1
    if value == 2:
        return 0
    if value == 3:
        return -1
    # sure we can do this with math, but this is fine for now


def gap_penalty(k):
    return 2 * k


def search_back_in_column(k, i, j):
    if i - k == 0:
        return 0
    return max(H[i - k][j] - gap_penalty(k), search_back_in_column(k + 1, i, j))


def search_back_in_row(k, i, j):
    if j - k == 0:
        return 0
    return max(H[i][j - k] - gap_penalty(k), search_back_in_row(k + 1, i, j))


H = np.zeros(shape=(len(pattern) + 1, len(string) + 1))

for i in range(1, len(pattern) + 1):
    for j in range(1, len(string) + 1):
        H[i][j] = max(
            H[i - 1][j - 1] + sub_matrix(string[j - 1], pattern[i - 1]),
            search_back_in_column(1, i, j),
            search_back_in_row(1, i, j),
            0
        )

print(H)
