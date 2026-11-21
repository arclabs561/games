/* Search sizes of every set of seven distinct nuts with one solution, by listing rings.
 *
 * The search and the measure are those of difficulty.py (see its docstring); this is a second
 * implementation in C, fast enough to visit every set. Every one-solution set has exactly one
 * ring, and renaming the numbers gives every center the same distribution of search sizes, so
 * it is enough to visit the rings around C0 = (1, 2, 3, 4, 5, 6): for each edge sequence and
 * each choice of middles (as in exact.py), build the ring, keep it if its seven nuts are
 * distinct, count the solutions of its set, and if there is one, add the set's two search
 * sizes to their histograms.
 *
 *   difficulty all [part parts]   histograms over the rings of edge sequences part, part + parts, ...
 *   difficulty sizes o1 .. o7     calls with look-ahead, calls plain and solutions of one set
 *                                  (each o is six digits, e.g. 123456)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXSIZE 4096

typedef struct {
    int order[6];  /* numbers 1..6 read counterclockwise */
    int left[7];   /* left[v]: the number before v */
    int right[7];  /* right[v]: the number after v */
} Nut;

static void make(Nut *n, const int *order) {
    for (int i = 0; i < 6; i++) n->order[i] = order[i];
    for (int i = 0; i < 6; i++) {
        n->left[order[i]] = order[(i + 5) % 6];
        n->right[order[i]] = order[(i + 1) % 6];
    }
}

static long calls, sols;
static const Nut *pool[7];
static int faces[6];
static int look;

static void place(int used, int k, int prev, int first) {
    int middle = faces[k - 1];
    const Nut *c = pool[0];
    int left = c->left[middle], right = c->right[middle];
    (void)c;
    for (int j = 0; j < 7; j++) {
        if (used >> j & 1) continue;
        int a = pool[j]->left[middle], b = pool[j]->right[middle];
        if (look && a == right) continue;
        if (prev < 0) {
            if (look && b == left) continue;
        } else if (b != prev) {
            continue;
        }
        calls++;
        if (k == 6) {
            sols += a == pool[first]->right[faces[0]];
        } else {
            place(used | 1 << j, k + 1, a, k == 1 ? j : first);
        }
    }
}

/* One complete search from side s of every center: as difficulty.search on turned orders. */
static void search(const Nut *set, int s, int look_ahead) {
    look = look_ahead;
    calls = 1;
    sols = 0;
    for (int c = 0; c < 7; c++) {
        calls++;
        /* the center goes first in pool, the others after it, in the given order */
        pool[0] = &set[c];
        int m = 1;
        for (int j = 0; j < 7; j++)
            if (j != c) pool[m++] = &set[j];
        for (int i = 0; i < 6; i++) faces[i] = set[c].order[(i + s) % 6];
        place(1, 1, -1, -1);
    }
}

static long size(const Nut *set, int look_ahead) {
    long total = 0;
    for (int s = 0; s < 6; s++) {
        search(set, s, look_ahead);
        total += calls;
    }
    return total;
}

/* A nut's index among the 120: the rank of its order turned to start with 1. */
static int nut_id(const int *order) {
    int i0 = 0;
    while (order[i0] != 1) i0++;
    int rest[5];
    for (int i = 0; i < 5; i++) rest[i] = order[(i0 + 1 + i) % 6];
    /* rank of the permutation rest of 2..6 in lexicographic order */
    int id = 0, fact[5] = {24, 6, 2, 1, 1};
    for (int i = 0; i < 5; i++) {
        int smaller = 0;
        for (int j = i + 1; j < 5; j++) smaller += rest[j] < rest[i];
        id += smaller * fact[i];
    }
    return id;
}

static const int PERMS[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

static int all(int part, int parts) {
    static long hist[2][MAXSIZE];
    int e[6], ne = 0;
    long rings = 0, distinct = 0, one = 0, by_sols[64] = {0};
    int c0[6] = {1, 2, 3, 4, 5, 6};
    int c0id = nut_id(c0);
    for (int code = 0; code < 46656; code++) {
        int x = code;
        for (int i = 0; i < 6; i++) { e[i] = x % 6 + 1; x /= 6; }
        int ok = 1;
        for (int i = 0; i < 6 && ok; i++) {
            int prev = e[(i + 5) % 6];
            if (e[i] == i + 1 || e[i] == (i + 1) % 6 + 1 || e[i] == prev) ok = 0;
        }
        if (!ok) continue;
        if (ne++ % parts != part) continue;
        /* the three numbers of each outer nut besides its center number and two edges */
        int rest[6][3];
        for (int i = 0; i < 6; i++) {
            int m = 0, before = e[(i + 5) % 6], after = e[i];
            for (int v = 1; v <= 6; v++)
                if (v != i + 1 && v != before && v != after) rest[i][m++] = v;
        }
        Nut set[7];
        make(&set[0], c0);
        for (int mids = 0; mids < 46656; mids++) {
            int y = mids, ids[7], order[6];
            ids[0] = c0id;
            int dup = 0;
            for (int i = 0; i < 6; i++) {
                const int *p = PERMS[y % 6];
                y /= 6;
                order[0] = i + 1;
                order[1] = e[(i + 5) % 6];
                order[2] = rest[i][p[0]];
                order[3] = rest[i][p[1]];
                order[4] = rest[i][p[2]];
                order[5] = e[i];
                make(&set[i + 1], order);
                ids[i + 1] = nut_id(order);
                for (int j = 0; j <= i; j++) dup |= ids[j] == ids[i + 1];
            }
            rings++;
            if (dup) continue;
            distinct++;
            search(set, 0, 1);
            by_sols[sols < 63 ? sols : 63]++;
            if (sols != 1) continue;
            one++;
            for (int l = 0; l < 2; l++) {
                long t = size(set, 1 - l);
                if (t >= MAXSIZE) { fprintf(stderr, "size %ld too large\n", t); return 1; }
                hist[l][t]++;
            }
        }
    }
    printf("edge_sequences %d\nrings %ld\ndistinct %ld\none %ld\n", ne, rings, distinct, one);
    printf("solutions");
    for (int k = 0; k < 64; k++) printf(" %ld", by_sols[k]);
    printf("\n");
    for (int l = 0; l < 2; l++) {
        printf("%s", l == 0 ? "look_ahead" : "plain");
        for (int t = 0; t < MAXSIZE; t++)
            if (hist[l][t]) printf(" %d:%ld", t, hist[l][t]);
        printf("\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "all") == 0) {
        int part = argc > 3 ? atoi(argv[2]) : 0, parts = argc > 3 ? atoi(argv[3]) : 1;
        return all(part, parts);
    }
    if (argc == 9 && strcmp(argv[1], "sizes") == 0) {
        Nut set[7];
        for (int i = 0; i < 7; i++) {
            int order[6];
            for (int j = 0; j < 6; j++) order[j] = argv[2 + i][j] - '0';
            make(&set[i], order);
        }
        long a = size(set, 1), b = size(set, 0);
        search(set, 0, 1);
        printf("%ld %ld %ld\n", a, b, sols);
        return 0;
    }
    fprintf(stderr, "usage: difficulty all [part parts] | difficulty sizes o1 .. o7\n");
    return 2;
}
