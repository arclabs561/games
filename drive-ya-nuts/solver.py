r"""
Solutions to Drive-Ya-Nuts and its variations.

There are 7 nuts, each of which can be rotated in 6 ways. So there are 6^7 ~=
1e5 candidate solutions, since the nuts are unique.

We will yield solutions by iteratively placing nuts along the following
ordered path in a DFS fashion.

        ___
       /   \
   .--<  5  >--.
  / 6  \___/ 4  \
  \    /   \    /
   >--<  1  >--<
  / 7  \___/ 3  \
  \    /   \    /
   `--<  2  >--`
       \___/

We call the ordered sequence of a nut's side values its order. The first value
of an order is defined in the original game, our method doesn't assume it.
However, we do assume that the first nut in the path (nut 1 above), is placed
such that its first value is facing toward hexagon 2.

We also consider solutions in the space of all possible 7 nuts with unique
permuted values. For a nut with 6 unique values, one for each side, there are
6!  permutations... But 6 rotations create equivalence classes of size 6, so
the size of class representatives is 6!/6 = 5!. Now, there does exist a fast
method [1] for generating these equivalence classes, but with only 6 sides it
will be easy enough to dedup with `deque.rotate`.

If we also assume that we use 7 unique nuts, then there are 5! choose 7 ~= 1e10
possible Drive-Ya-Nuts boards. Whats the distribution of number of solutions
over all these boards?

[1] http://www.cis.uoguelph.ca/~sawada/papers/alph.pdf
"""
from collections import Counter, deque
from itertools import permutations
from math import comb
from random import sample


class Nut(object):
    """Simple data structure for faster retrieval of adjacent nut values."""
    def __init__(self, order):
        self.order = tuple(order)
        self.mapping = self.build_mapping(self.order)
        self._canon = None
        self._hash = None

    def __eq__(self, other):
        if not isinstance(other, Nut):
            return NotImplemented
        return self.canonical_order() == other.canonical_order()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.canonical_order())
        return self._hash

    def canonical_order(self):
        if self._canon is None:
            order = deque(self.order)
            min_ = min(self.order)
            while order[0] != min_:
                order.rotate(1)
            self._canon = tuple(order)
        return self._canon

    def __repr__(self):
        return 'Nut({})'.format(self.order)

    @staticmethod
    def build_mapping(order):
        mapping = {}
        for i, val in enumerate(order):
            mapping[val] = (order[i - 1],
                            order[(i + 1) % len(order)])
        return mapping

    def left(self, val):
        return self.mapping[val][0]

    def right(self, val):
        return self.mapping[val][1]


# Here we define the original game nuts, lexographically, whose orders are read
# counter-clockwise, starting from 1.
#
# http://www.hasbro.com/common/instruct/DriveYaNuts.PDF
nuts = {Nut(order) for order in [
    (1, 2, 3, 4, 5, 6),
    (1, 2, 5, 6, 3, 4),
    (1, 3, 5, 2, 4, 6),
    (1, 3, 5, 4, 2, 6),
    (1, 4, 2, 3, 5, 6),
    (1, 5, 3, 2, 6, 4),
    (1, 6, 5, 4, 3, 2)
]}


def recur(pool, path, prev):
    """Yield solutions to Drive-Ya-Nuts.

    `(path, prev)` should be initialized to `([], None)` respectively to yield
    solutions completely. And it's intended that `len(pool) == 7`, and
    `all(len(nut.order) == 6 for nut in pool)`.

    Args:
        pool: `set` of `Nut`s
        path: list of ordered nuts in candidate solution
        prev: previous nut value that will be left-adjacent
    Yields:
        Valid path solutions, which are each lists of `Nut`s.
    """
    if not pool:
        yield path
        return

    if not path:
        for nut in pool:
            yield from recur(pool - {nut}, [nut], None)
        return

    center = path[0]
    k = len(path)
    middle = center.order[k - 1]  # center nut value facing candidate nut
    left, right = center.left(middle), center.right(middle)

    for nut in pool:
        if nut.left(middle) == right:
            continue
        if prev is None and nut.right(middle) == left:
            continue
        if prev is not None and nut.right(middle) != prev:
            continue
        yield from recur(pool - {nut},         # remaining pool
                         path.copy() + [nut],  # path with candidate nut
                         nut.left(middle))     # left of nut will be prev


if __name__ == '__main__':
    def print_header(text):
        print('\n\033[1m\033[31m\033[4m{}\033[0m'.format(text))

    # Original solutions
    print_header('Original solutions')
    print(list(recur(nuts, [], None)))

    # Approximate distribution of solutions.
    #
    # We use Hoeffding's inequality to be 0.95 confident that we are 0.01
    # within true probabilities. That is n >= 1/(2*epsilon^2) * log(2/alpha) ~=
    # 18444.
    trials = 18444
    all_nuts = sorted(
        {Nut(perm) for perm in permutations(range(1, 7))},
        key=lambda n: n.canonical_order(),
    )
    counts = Counter()

    for _ in range(trials):
        nuts = set(sample(all_nuts, 7))
        n_sols = sum(1 for _ in recur(nuts, [], None))
        counts[n_sols] += 1

    print_header('Distribution of number of solutions')
    total = comb(len(all_nuts), 7)
    for k, v in counts.most_common():
        f = v/trials
        print('{:<4}{:<10.2e}{:<8.1%}{}'.format(
            k, f*total, f, '='*int(f*40)))
