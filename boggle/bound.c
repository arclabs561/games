/* Branch and bound over classes of Boggle boards, after Vanderkam (2025, arXiv 2507.02117;
   the 2009 bounds his paper describes).

   A class gives each cell a set of letters and stands for every board that takes one letter
   from each set. Two upper bounds hold for every board in a class:

     max:   a depth-first walk over paths, as in scoring, where at each step the walk takes the
            letter of the cell whose subtree is worth most. Each path is counted, so a word
            spelled by two paths counts twice; a board's score counts each word once, and its
            own walk is never worth more than the walk that takes the best letter everywhere.
     union: the points of every different word that some board of the class shows.

   A class whose smaller bound is below the threshold holds no board that reaches it. Any
   other class is split into one class per letter of one cell: the cell with the most
   neighbors among those with more than one letter (the first such cell on a tie). A class of
   single letters is a board, and is scored.

   The search starts from the classes that give each cell one part of a partition of the
   letters (for example "aeiosuy" and the rest), one class for each assignment up to the
   symmetries of the grid: an assignment is searched only if no rotation or reflection of it
   comes first in order. Every board is a rotation or reflection of a board in a searched
   class, with the same score.

   A word is a path of at least three letters through neighboring cells (sideways or
   diagonally), each cell used once. The letter q stands for the face Qu: the word list keeps
   only words whose every q is followed by u, and walks "qu" as one step.

   Commands:
     solve <words> <rows> <cols> <threshold> <partition> <shard> <shards> [<certificate>]
         Searches the classes numbered shard, shard + shards, ... in the order of the
         assignments (numbered from 0, counting only those searched). Prints "board <letters>
         <points>" for every board that scores at least the threshold, then one line of counts.
         With a certificate path, writes the search tree there in depth-first order: "c" and
         the part of each cell for a starting class, then "s <cell>" for a split (followed by
         one subtree per letter of that cell, in alphabetical order), "p <bound>" for a pruned
         class and "b <points>" for a board.
     bound <words> <rows> <cols> <cell sets...>
         Prints the max and union bounds of one class, cells given as letter sets row by row.
     score <words> <rows> <cols> <letters>
         Prints the points and number of words of one board.
*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int32_t next[26]; int16_t points; uint32_t mark; } Node;
static Node *trie;
static int nnodes, capnodes;
static uint32_t stamp;

static int new_node(void) {
    if (nnodes == capnodes) {
        capnodes = capnodes ? 2 * capnodes : 1 << 16;
        trie = realloc(trie, (size_t)capnodes * sizeof *trie);
        if (!trie) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memset(&trie[nnodes], 0, sizeof *trie);
    return nnodes++;
}

static int word_points(int len) {
    return len < 3 ? 0 : len <= 4 ? 1 : len == 5 ? 2 : len == 6 ? 3 : len == 7 ? 5 : 11;
}

/* Insert each word of at least three letters whose letters are a to z and whose every q is
   followed by u; "qu" becomes the single step q. */
static void load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    new_node();
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int len = 0, ok = 1;
        while (line[len] && line[len] != '\n' && line[len] != '\r') {
            if (line[len] < 'a' || line[len] > 'z') ok = 0;
            if (line[len] == 'q' && line[len + 1] != 'u') ok = 0;
            len++;
        }
        if (!ok || len < 3) continue;
        int node = 0;
        for (int i = 0; i < len; i++) {
            int c = line[i] - 'a';
            if (!trie[node].next[c]) { int m = new_node(); trie[node].next[c] = m; }
            node = trie[node].next[c];
            if (line[i] == 'q') i++; /* the u of qu */
        }
        trie[node].points = (int16_t)word_points(len);
    }
    fclose(f);
}

enum { MAXCELLS = 16 };
static int rows, cols, ncells;
static int nbr[MAXCELLS][8], nnbr[MAXCELLS];

static void grid(int r, int c) {
    rows = r, cols = c, ncells = r * c;
    if (ncells > MAXCELLS) { fprintf(stderr, "grid too large\n"); exit(2); }
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            int k = i * c + j;
            nnbr[k] = 0;
            for (int di = -1; di <= 1; di++)
                for (int dj = -1; dj <= 1; dj++)
                    if ((di || dj) && i + di >= 0 && i + di < r && j + dj >= 0 && j + dj < c)
                        nbr[k][nnbr[k]++] = (i + di) * c + j + dj;
        }
}

static uint32_t cell[MAXCELLS]; /* letter sets, bit c for letter 'a' + c */

static long max_walk(int k, int node, int used) {
    used |= 1 << k;
    long best = 0;
    for (uint32_t m = cell[k]; m; m &= m - 1) {
        int child = trie[node].next[__builtin_ctz(m)];
        if (!child) continue;
        long v = trie[child].points;
        for (int i = 0; i < nnbr[k]; i++)
            if (!(used >> nbr[k][i] & 1)) v += max_walk(nbr[k][i], child, used);
        if (v > best) best = v;
    }
    return best;
}

static long max_bound(void) {
    long total = 0;
    for (int k = 0; k < ncells; k++) total += max_walk(k, 0, 0);
    return total;
}

/* Every word some board of the class shows, each counted once (a stamp on its node). */
static long union_walk(int k, int node, int used) {
    used |= 1 << k;
    long total = 0;
    for (uint32_t m = cell[k]; m; m &= m - 1) {
        int child = trie[node].next[__builtin_ctz(m)];
        if (!child) continue;
        if (trie[child].points && trie[child].mark != stamp) {
            trie[child].mark = stamp;
            total += trie[child].points;
        }
        for (int i = 0; i < nnbr[k]; i++)
            if (!(used >> nbr[k][i] & 1)) total += union_walk(nbr[k][i], child, used);
    }
    return total;
}

static long union_bound(void) {
    stamp++;
    long total = 0;
    for (int k = 0; k < ncells; k++) total += union_walk(k, 0, 0);
    return total;
}

static long score(void) { return union_bound(); } /* for a class of single letters */

static long count_words(void) { /* the words on a board, after score() set the stamps */
    long n = 0;
    for (int i = 0; i < nnodes; i++) n += trie[i].points && trie[i].mark == stamp;
    return n;
}

static long threshold;
static FILE *cert;
static long nclasses, nodes, pruned, boards, found;

static void board_letters(char *out) {
    for (int k = 0; k < ncells; k++) out[k] = (char)('a' + __builtin_ctz(cell[k]));
    out[ncells] = 0;
}

static void solve(void) {
    nodes++;
    int split = -1;
    for (int k = 0; k < ncells; k++)
        if (__builtin_popcount(cell[k]) > 1 && (split < 0 || nnbr[k] > nnbr[split])) split = k;
    if (split < 0) {
        long s = score();
        boards++;
        if (cert) fprintf(cert, "b %ld\n", s);
        if (s >= threshold) {
            char b[MAXCELLS + 1];
            board_letters(b);
            printf("board %s %ld\n", b, s);
            found++;
        }
        return;
    }
    long b = max_bound();
    if (b >= threshold) {
        long u = union_bound();
        if (u < b) b = u;
    }
    if (b < threshold) {
        pruned++;
        if (cert) fprintf(cert, "p %ld\n", b);
        return;
    }
    uint32_t all = cell[split];
    if (cert) fprintf(cert, "s %d\n", split);
    for (uint32_t m = all; m; m &= m - 1) {
        cell[split] = m & -m;
        solve();
    }
    cell[split] = all;
}

/* The image of cell k under symmetry g of the grid (4 for a rectangle, 8 for a square). */
static int image(int g, int k) {
    int i = k / cols, j = k % cols, n = rows;
    switch (g) {
    case 0: return k;
    case 1: return i * cols + (cols - 1 - j);                    /* mirror left to right */
    case 2: return (rows - 1 - i) * cols + j;                    /* mirror top to bottom */
    case 3: return (rows - 1 - i) * cols + (cols - 1 - j);       /* half turn */
    case 4: return j * n + i;                                    /* transpose */
    case 5: return j * n + (n - 1 - i);                          /* quarter turn */
    case 6: return (n - 1 - j) * n + i;                          /* three quarter turns */
    default: return (n - 1 - j) * n + (n - 1 - i);               /* anti-transpose */
    }
}

static int canonical(const int *part) {
    int nsym = rows == cols ? 8 : 4;
    for (int g = 1; g < nsym; g++) {
        int t[MAXCELLS];
        for (int k = 0; k < ncells; k++) t[image(g, k)] = part[k];
        for (int k = 0; k < ncells; k++) {
            if (t[k] < part[k]) return 0;
            if (t[k] > part[k]) break;
        }
    }
    return 1;
}

static uint32_t letters_mask(const char *s) {
    uint32_t m = 0;
    for (; *s; s++)
        if (*s >= 'a' && *s <= 'z') m |= 1u << (*s - 'a');
    return m;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: see the comment at the top of bound.c\n"); return 2; }
    load(argv[2]);
    grid(atoi(argv[3]), atoi(argv[4]));
    if (!strcmp(argv[1], "score") && argc == 6) {
        for (int k = 0; k < ncells; k++) cell[k] = 1u << (argv[5][k] - 'a');
        long s = score();
        printf("%ld %ld\n", s, count_words());
        return 0;
    }
    if (!strcmp(argv[1], "bound") && argc == 5 + ncells) {
        for (int k = 0; k < ncells; k++) cell[k] = letters_mask(argv[5 + k]);
        long m = max_bound(), u = union_bound();
        printf("%ld %ld\n", m, u);
        return 0;
    }
    if (!strcmp(argv[1], "solve") && (argc == 9 || argc == 10)) {
        threshold = atol(argv[5]);
        long shard = atol(argv[7]), shards = atol(argv[8]);
        uint32_t parts[26];
        int nparts = 0;
        char *spec = strdup(argv[6]);
        for (char *tok = strtok(spec, " "); tok; tok = strtok(NULL, " ")) parts[nparts++] = letters_mask(tok);
        if (argc == 10 && !(cert = fopen(argv[9], "w"))) { fprintf(stderr, "cannot write\n"); return 2; }
        int part[MAXCELLS] = {0};
        long index = 0;
        for (;;) {
            if (canonical(part) && index++ % shards == shard) {
                nclasses++;
                for (int k = 0; k < ncells; k++) cell[k] = parts[part[k]];
                if (cert) {
                    fprintf(cert, "c");
                    for (int k = 0; k < ncells; k++) fprintf(cert, " %d", part[k]);
                    fprintf(cert, "\n");
                }
                solve();
                if (nclasses % 1000 == 0) fprintf(stderr, "%ld classes, %ld nodes\n", nclasses, nodes);
            }
            int k = ncells - 1;
            while (k >= 0 && part[k] == nparts - 1) part[k--] = 0;
            if (k < 0) break;
            part[k]++;
        }
        printf("classes %ld nodes %ld pruned %ld boards %ld found %ld\n", nclasses, nodes, pruned,
               boards, found);
        if (cert) fclose(cert);
        return 0;
    }
    fprintf(stderr, "usage: see the comment at the top of bound.c\n");
    return 2;
}
