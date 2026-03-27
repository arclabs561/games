from collections import Counter
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pyprind
import scipy.stats as stats


def roll_dice(n=1):
    return (randint(1, 6) for _ in range(n))


def tenzi_turns():
    initial = Counter(roll_dice(10))
    match, count = initial.most_common(1)[0]
    turns = 1
    while count < 10:
        rerolled = Counter(roll_dice(10 - count))
        count += rerolled[match]
        turns += 1
    return turns


def tenzi_sample(n):
    pbar = pyprind.ProgPercent(n)
    samples = []
    for _ in range(n):
        samples.append(tenzi_turns())
        pbar.update()
    return samples


if __name__ == "__main__":
    data = tenzi_sample(10**4)
    plt.hist(data, bins=60, density=True, color="w")
    shape, loc, scale = stats.gamma.fit(data, floc=1)
    print(shape, loc, scale)
    rv = stats.gamma(shape, loc, scale)
    x = np.linspace(0, 60)
    plt.plot(x, rv.pdf(x))
    plt.show()
