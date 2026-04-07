import numpy as np
import math


def select(input_array, size):
    input_array = np.asarray(input_array)
    n = len(input_array)

    order = np.argsort(input_array)
    sorted_input = input_array[order]

    target = np.linspace(sorted_input[0], sorted_input[-1], size)
    j = np.searchsorted(sorted_input, target)

    left = np.clip(j - 1, 0, n - 1)
    right = np.clip(j, 0, n - 1)

    dist_left = np.abs(sorted_input[left] - target)
    dist_right = np.abs(sorted_input[right] - target)

    selected = np.where(dist_right < dist_left, right, left)

    used = np.zeros(n, dtype=bool)
    result = np.empty(size, dtype=int)

    for i, idx in enumerate(selected):
        if not used[idx]:
            used[idx] = True
            result[i] = idx
            continue

        l = idx - 1
        r = idx + 1

        while True:
            if l >= 0 and not used[l]:
                used[l] = True
                result[i] = l
                break
            if r < n and not used[r]:
                used[r] = True
                result[i] = r
                break
            l -= 1
            r += 1

    return np.sort(order[result])


def approx(N):
    n = int(math.sqrt(N))

    is_even = (n % 2 == 0)
    l = n // 2

    idx = np.arange(1, l + 1)
    if is_even:
        layersizes = 8 * idx - 4
    else:
        layersizes = np.concatenate([[1], 8 * idx])

    nn = n ** 2
    return nn, n, layersizes[::-1]