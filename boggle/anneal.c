/* Search for high-scoring square Boggle boards: simulated annealing, then a full climb.

   The word list is read into a trie; a board is scored by a depth-first search from every
   square, as board.get_words does, and each word counts once (a stamp on its trie node).
   The face "qu" is written q on a board and walks q then u in the trie.

   Commands, on the command line:
     score <words> <n> <letters>          the points and the number of words on one board
     climb <words> <n> <letters>          climb from one board (below) and print the top
     anneal <words> <n> <seed> <runs> <steps> <t0> <t1>
                                          `runs` searches from random boards. Each search
                                          makes `steps` changes at temperatures falling
                                          geometrically from t0 to t1. A change sets one
                                          square to a random letter or swaps two squares. A
                                          change that lowers the score by d is kept with
                                          chance exp(-d / t). Then the best board seen is
                                          climbed: every change of one square to any of the
                                          26 faces, and every swap, is tried in order, and
                                          the first that raises the score is kept, until
                                          none does. One line per search:
                                          "run <seed> <i> <points> <words> <board>".
   Random letters come from Norvig's string of cube letters (board.NORVIG_LETTERS), the
   random numbers from splitmix64 seeded with the seed and the run number.
*/
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *LETTERS = "AABCDEEEGHIILMNOOPRSTUYAABCDEEEGHIILMNOOPRSTUYAABCDEEEGHIILMNOOPRSTUY"
                             "AADEFFIJKKLLNNQRSSTTUVVWWXZ";

typedef struct { int32_t next[26]; int16_t points; uint32_t stamp; } Node;
static Node *trie;
static int nnodes, capnodes;

static int new_node(void) {
    if (nnodes == capnodes) {
        capnodes = capnodes ? 2 * capnodes : 1 << 16;
        trie = realloc(trie, (size_t)capnodes * sizeof *trie);
        if (!trie) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memset(&trie[nnodes], 0, sizeof *trie);
    return nnodes++;
}

static int points(int len) {
    return len < 3 ? 0 : len <= 4 ? 1 : len == 5 ? 2 : len == 6 ? 3 : len == 7 ? 5 : 11;
}

static void load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    new_node();
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int len = 0, ok = 1;
        while (line[len] && line[len] != '\n' && line[len] != '\r') {
            if (line[len] < 'a' || line[len] > 'z') ok = 0;
            len++;
        }
        if (!ok || len < 3) continue;
        int node = 0;
        for (int i = 0; i < len; i++) {
            int c = line[i] - 'a';
            if (!trie[node].next[c]) { int k = new_node(); trie[node].next[c] = k; }
            node = trie[node].next[c];
        }
        trie[node].points = (int16_t)points(len);
    }
    fclose(f);
}

static int n, cells;
static int nbr[64][8], nnbr[64];
static uint32_t stamp;
static int words, total;

static void walk(const int *b, int k, uint64_t used, int node) {
    int c = b[k];
    node = trie[node].next[c];
    if (!node) return;
    if (c == 'q' - 'a') { node = trie[node].next['u' - 'a']; if (!node) return; }
    used |= 1ULL << k;
    if (trie[node].points && trie[node].stamp != stamp) {
        trie[node].stamp = stamp;
        total += trie[node].points;
        words++;
    }
    for (int i = 0; i < nnbr[k]; i++)
        if (!(used >> nbr[k][i] & 1)) walk(b, nbr[k][i], used, node);
}

static int score(const int *b, int *nwords) {
    stamp++;
    words = total = 0;
    for (int k = 0; k < cells; k++) walk(b, k, 0, 0);
    if (nwords) *nwords = words;
    return total;
}

static uint64_t rng;
static uint64_t next(void) {
    uint64_t z = (rng += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static double uniform(void) { return (next() >> 11) * (1.0 / 9007199254740992.0); }
static int letter(void) { return LETTERS[next() % strlen(LETTERS)] - 'A'; }

static void show(const int *b, char *out) {
    for (int k = 0; k < cells; k++) out[k] = (char)('a' + b[k]);
    out[cells] = 0;
}

/* Climb until no change of one square and no swap raises the score. */
static int climb(int *b, int best) {
    for (int better = 1; better;) {
        better = 0;
        for (int k = 0; k < cells; k++)
            for (int c = 0; c < 26; c++) {
                int old = b[k];
                if (c == old) continue;
                b[k] = c;
                int s = score(b, 0);
                if (s > best) { best = s; better = 1; } else b[k] = old;
            }
        for (int i = 0; i < cells; i++)
            for (int j = i + 1; j < cells; j++) {
                if (b[i] == b[j]) continue;
                int t = b[i]; b[i] = b[j]; b[j] = t;
                int s = score(b, 0);
                if (s > best) { best = s; better = 1; } else { b[j] = b[i]; b[i] = t; }
            }
    }
    return best;
}

static void geometry(void) {
    cells = n * n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int k = n * i + j;
            nnbr[k] = 0;
            for (int di = -1; di <= 1; di++)
                for (int dj = -1; dj <= 1; dj++)
                    if ((di || dj) && i + di >= 0 && i + di < n && j + dj >= 0 && j + dj < n)
                        nbr[k][nnbr[k]++] = n * (i + di) + j + dj;
        }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: see the comment at the top\n"); return 1; }
    load(argv[2]);
    n = atoi(argv[3]);
    if (n < 2 || n > 7) { fprintf(stderr, "n from 2 to 7\n"); return 1; }
    geometry();
    int b[64], keep[64];
    char text[65];
    if (!strcmp(argv[1], "score") && argc == 5) {
        if ((int)strlen(argv[4]) != cells) { fprintf(stderr, "need %d letters\n", cells); return 1; }
        for (int k = 0; k < cells; k++) b[k] = (argv[4][k] | 32) - 'a';
        int w, s = score(b, &w);
        printf("%d %d\n", s, w);
        return 0;
    }
    if (!strcmp(argv[1], "climb") && argc == 5) {
        if ((int)strlen(argv[4]) != cells) { fprintf(stderr, "need %d letters\n", cells); return 1; }
        for (int k = 0; k < cells; k++) b[k] = (argv[4][k] | 32) - 'a';
        int w, s = climb(b, score(b, 0));
        score(b, &w);
        show(b, text);
        printf("%d %d %s\n", s, w, text);
        return 0;
    }
    if (!strcmp(argv[1], "anneal") && argc == 9) {
        uint64_t seed = strtoull(argv[4], 0, 10);
        int runs = atoi(argv[5]);
        long steps = atol(argv[6]);
        double t0 = atof(argv[7]), t1 = atof(argv[8]);
        for (int run = 0; run < runs; run++) {
            rng = seed * 1000003ULL + (uint64_t)run;
            for (int k = 0; k < cells; k++) b[k] = letter();
            int cur = score(b, 0), best = cur;
            memcpy(keep, b, sizeof b);
            for (long s = 0; s < steps; s++) {
                double t = t0 * pow(t1 / t0, (double)s / (double)steps);
                int i = (int)(next() % cells), j = -1, old = b[i];
                if (next() & 1) b[i] = letter();
                else {
                    j = (int)(next() % cells);
                    b[i] = b[j]; b[j] = old;
                }
                int sc = score(b, 0);
                if (sc >= cur || uniform() < exp((sc - cur) / t)) {
                    cur = sc;
                    if (cur > best) { best = cur; memcpy(keep, b, sizeof b); }
                } else if (j < 0) b[i] = old;
                else { b[j] = b[i]; b[i] = old; }
            }
            memcpy(b, keep, sizeof b);
            best = climb(b, best);
            int w, s = score(b, &w);
            show(b, text);
            printf("run %llu %d %d %d %s\n", (unsigned long long)seed, run, s, w, text);
            fflush(stdout);
        }
        return 0;
    }
    fprintf(stderr, "unknown command\n");
    return 1;
}
