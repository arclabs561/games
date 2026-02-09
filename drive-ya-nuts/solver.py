r"""
Solutions to Drive-Ya-Nuts and its variations.

There are 7 nuts, each of which can be rotated in 6 ways. So there are 6^7 ~=
3e5 rotation assignments for a fixed placement. If you also permute which nut
goes in which position, a naive brute force search is 7! * 6^7 ~= 1e9.

In this solver we model each nut by the cyclic adjacency of its values. Once
the center-facing value is fixed for a position, an outer nut's rotation is
determined, so the search is much closer to permuting nuts with pruning than it
is to exploring all 6^7 rotations.

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

If we also assume that we use 7 unique nuts, then there are C(5!, 7) =
C(120, 7) ~= 6e10 possible sets of pieces. What's the distribution of number of
solutions over all these boards?

[1] http://www.cis.uoguelph.ca/~sawada/papers/alph.pdf
"""
import argparse
from collections import Counter, deque
from itertools import permutations
from math import ceil, comb, log
import random
import time


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


# Here we define the original game nuts, lexicographically, whose orders are read
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
        # Ring closure: nut 7 must match nut 2.
        #
        # The recursion enforces matches between consecutive outer nuts
        # (2-3-4-5-6-7). When the pool is exhausted, `prev` is the value on the
        # final nut (nut 7) that should be adjacent to the first outer nut.
        if len(path) == 7:
            center = path[0]
            nut2 = path[1]
            middle2 = center.order[0]
            if prev != nut2.right(middle2):
                return
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


def hoeffding_trials(*, epsilon: float, alpha: float, m: int = 1) -> int:
    """Sufficient n for Hoeffding + (optional) union bound.

    For a fixed k, let p_k = P(n_sols == k) over uniformly sampled boards and
    let p̂_k be the empirical frequency over n i.i.d. trials. Hoeffding gives:

        P(|p̂_k - p_k| >= epsilon) <= 2 exp(-2 n epsilon^2).

    If you want this to hold simultaneously for m different k values, apply a
    union bound:

        P(max_k |p̂_k - p_k| >= epsilon) <= 2 m exp(-2 n epsilon^2).

    Solving for n with RHS <= alpha yields:

        n >= (1 / (2 epsilon^2)) * log(2 m / alpha).
    """
    if epsilon <= 0:
        raise ValueError('epsilon must be > 0')
    if not (0 < alpha < 1):
        raise ValueError('alpha must be in (0, 1)')
    if m <= 0:
        raise ValueError('m must be > 0')
    return ceil((1 / (2 * epsilon**2)) * log((2 * m) / alpha))


def print_header(text):
    print('\n\033[1m\033[31m\033[4m{}\033[0m'.format(text))


def count_solutions(pool, *, stop_after=None):
    """Count solutions for a given 7-nut pool, with optional early stop.

    Args:
        pool: `set` of `Nut`s
        stop_after: if not None, stop once count exceeds this value
    """
    n = 0
    for _ in recur(pool, [], None):
        n += 1
        if stop_after is not None and n > stop_after:
            break
    return n


def format_nut_set(pool):
    """Deterministic, copy/pasteable representation of a 7-nut piece set."""
    ordered = sorted(pool, key=lambda n: n.canonical_order())
    return '[' + ', '.join(str(n.canonical_order()) for n in ordered) + ']'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--original',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Print solutions for the original 7-nut Hasbro puzzle.',
    )
    parser.add_argument(
        '--distribution',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Estimate the solution-count distribution over random piece sets.',
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=None,
        help=('Number of random piece sets to sample. If omitted, compute a '
              'sufficient value via Hoeffding using --epsilon/--alpha/--m.'),
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.01,
        help='Hoeffding absolute error target (per bin).',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Hoeffding failure probability (e.g. 0.05 for 95%% confidence).',
    )
    parser.add_argument(
        '--m',
        type=int,
        default=1,
        help=('Union-bound size (number of k values you want the bound to hold '
              'over). Use 1 for a single bin; for the full distribution over '
              'k in [0, 7!], m <= 5041 is safe.'),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed for reproducible sampling.',
    )
    parser.add_argument(
        '--sort-by',
        choices=['k', 'freq'],
        default='k',
        help='How to sort the printed distribution table.',
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=0,
        help='If >0, print a progress line every N trials.',
    )
    parser.add_argument(
        '--find-k',
        type=int,
        default=None,
        help=('Sample random piece sets until one has exactly K solutions; '
              'prints the set (and optionally some solutions).'),
    )
    parser.add_argument(
        '--max-tries',
        type=int,
        default=100000,
        help='Max random piece sets to try when using --find-k.',
    )
    parser.add_argument(
        '--print-found',
        type=int,
        default=1,
        help='How many solutions to print when --find-k succeeds (0 for none).',
    )
    args = parser.parse_args(argv)

    if not args.original and not args.distribution and args.find_k is None:
        parser.error('nothing to do (enable --original, --distribution, and/or --find-k)')

    needs_all_nuts = args.distribution or (args.find_k is not None)
    all_nuts = None
    if needs_all_nuts:
        all_nuts = sorted(
            {Nut(perm) for perm in permutations(range(1, 7))},
            key=lambda n: n.canonical_order(),
        )

    if args.original:
        print_header('Original solutions')
        sols = list(recur(nuts, [], None))
        sols.sort(key=lambda sol: tuple(n.canonical_order() for n in sol))
        print(f'{len(sols)} solution(s)')
        print(sols)

    if args.distribution:
        print_header('Distribution of number of solutions')
        assert all_nuts is not None
        total = comb(len(all_nuts), 7)

        trials = args.trials
        if trials is None:
            trials = hoeffding_trials(
                epsilon=args.epsilon,
                alpha=args.alpha,
                m=args.m,
            )
            print(f'Using trials={trials} (Hoeffding, epsilon={args.epsilon}, alpha={args.alpha}, m={args.m})')
        else:
            rec = hoeffding_trials(
                epsilon=args.epsilon,
                alpha=args.alpha,
                m=args.m,
            )
            print(f'Using trials={trials} (Hoeffding recommends >= {rec} for epsilon={args.epsilon}, alpha={args.alpha}, m={args.m})')

        rng = random.Random(args.seed)
        counts = Counter()
        t0 = time.perf_counter()

        for i in range(trials):
            piece_set = set(rng.sample(all_nuts, 7))
            n_sols = count_solutions(piece_set)
            counts[n_sols] += 1

            if args.progress_every and (i + 1) % args.progress_every == 0:
                dt = time.perf_counter() - t0
                print(f'... {i + 1}/{trials} piece sets ({dt:.1f}s elapsed)')

        dt = time.perf_counter() - t0
        print(f'Elapsed: {dt:.2f}s')

        n0 = counts.get(0, 0)
        p0 = n0 / trials
        p_solvable = 1 - p0
        p_unique = counts.get(1, 0) / trials
        e_sols = sum(k * v for k, v in counts.items()) / trials
        e_sols_given = (e_sols / p_solvable) if p_solvable else float('nan')

        print('')
        print(f'P(solvable): {p_solvable:.2%}')
        print(f'P(unique solution): {p_unique:.2%}')
        if p_solvable:
            print(f'P(unique | solvable): {(p_unique / p_solvable):.2%}')
        print(f'E[#solutions]: {e_sols:.4f}')
        if p_solvable:
            print(f'E[#solutions | solvable]: {e_sols_given:.4f}')
        print('')

        items = counts.most_common() if args.sort_by == 'freq' else sorted(counts.items())
        for k, v in items:
            f = v / trials
            print('{:<4}{:<10.2e}{:<8.1%}{}'.format(
                k, f * total, f, '=' * int(f * 40)))

    if args.find_k is not None:
        print_header(f'Find piece set with exactly {args.find_k} solution(s)')
        assert all_nuts is not None
        rng = random.Random(args.seed)
        target = args.find_k
        if target < 0:
            parser.error('--find-k must be >= 0')
        if args.max_tries <= 0:
            parser.error('--max-tries must be > 0')
        if args.print_found < 0:
            parser.error('--print-found must be >= 0')

        t0 = time.perf_counter()
        for i in range(args.max_tries):
            piece_set = set(rng.sample(all_nuts, 7))
            n_sols = count_solutions(piece_set, stop_after=target)
            if n_sols != target:
                continue

            dt = time.perf_counter() - t0
            print(f'Found after {i + 1} tries ({dt:.2f}s).')
            print('nuts =', format_nut_set(piece_set))
            print(f'n_sols = {n_sols}')
            if args.print_found and target > 0:
                print('')
                shown = 0
                for sol in recur(piece_set, [], None):
                    print(sol)
                    shown += 1
                    if shown >= args.print_found:
                        break
            break
        else:
            dt = time.perf_counter() - t0
            print(f'No match found after {args.max_tries} tries ({dt:.2f}s).')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
