/* Solver B: an exhaustive FreeCell solver written apart from Solver A (solver.py).

   A position is one byte per card: what lies directly beneath it. That is another card,
   BASE (the card is the bottom of a column), CELL (it is in a free cell) or HOME (it is on
   its foundation). The array is the same whatever the order of the columns and the cells, so
   it serves as the key of the position with no sorting. Columns in use are the cards on
   BASE; the rest are empty.

   The search is a depth-first search over single-card moves, by default with no autoplay. Every position
   reached is stored in a hash table and never expanded twice. With no budget the search
   either finds a won position or visits every reachable position.

   Input, one problem per line on stdin:
     ms <deal> <cells>                 a Microsoft deal (its own generator, below)
     deck <ranks> <suits> <columns> <cells> <card> <card> ... | ...
                                       an explicit layout; columns separated by '|',
                                       cards as numbers 4*(rank-1)+suit, bottom card first
   Output, one line per problem:
     won <states> <moves...>     each move "<card>><dest>" with dest h, c, e or a card number
     lost <states> <most cards home> <cards that never move...>
     unknown <states>
   Arguments, all optional: the budget (the most positions stored, 0 for no limit), a seed
   (0 for the fixed order; another number shuffles ties in the order of moves), then the
   words "fence", "auto" and "layers". With "fence", a lost problem is followed by its
   positions, one per line, each the 52 numbers of the array, and a line "end". With "auto",
   safe cards are played home after every move (see settle), which makes the stored
   positions fewer.

   With "layers" (which implies "auto" and ignores the budget and the seed), the positions
   are taken one foundation at a time (solve_layers below): only the positions with the
   foundations of the current layer are stored, and a position with more cards home waits in
   a list for its own layer. Cards never leave the foundations, so a layer is complete once
   every layer with fewer cards home is done. The output is as above, followed by one line
   "layer <clubs> <diamonds> <hearts> <spades> <positions>" per layer and a line "end".
*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NCARD = 52, BASE = 52, CELL = 53, HOME = 54, ABSENT = 55 };
#define KEYW 5 /* 64-bit words per packed key: 52 cards x 6 bits = 312 bits */

static int nranks, nsuits, ncols, ncells;
static int present[NCARD];
static int deck_size;

static int rk(int c) { return c / 4 + 1; }
static int st(int c) { return c % 4; }
static int red(int c) { return st(c) == 1 || st(c) == 2; }
static int stacks(int c, int onto) { return rk(onto) == rk(c) + 1 && red(onto) != red(c); }

typedef struct {
    uint8_t under[NCARD];
    uint8_t covered[NCARD]; /* 1 if some card lies on this one */
    int home[4], cells, bases, nhome;
} Pos;

/* ---- hash set of packed positions ---- */
static uint64_t *keys;   /* count * KEYW words */
static uint32_t *slots;  /* index + 1, 0 = empty */
static uint64_t nkeys, keycap, nslots;

static void pack(const Pos *p, uint64_t *k) {
    memset(k, 0, KEYW * 8);
    for (int c = 0; c < NCARD; c++) {
        uint64_t v = p->under[c] & 63;
        int bit = 6 * c;
        k[bit / 64] |= v << (bit % 64);
        if (bit % 64 > 58) k[bit / 64 + 1] |= v >> (64 - bit % 64);
    }
}
static uint64_t hash(const uint64_t *k) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < KEYW; i++) {
        h ^= k[i];
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 31;
    }
    return h;
}
static void grow(void) {
    uint64_t n2 = nslots ? nslots * 2 : 1 << 16;
    uint32_t *s2 = calloc(n2, sizeof *s2);
    if (!s2) { fprintf(stderr, "out of memory\n"); exit(2); }
    for (uint64_t i = 0; i < nkeys; i++) {
        uint64_t j = hash(keys + i * KEYW) & (n2 - 1);
        while (s2[j]) j = (j + 1) & (n2 - 1);
        s2[j] = (uint32_t)(i + 1);
    }
    free(slots);
    slots = s2;
    nslots = n2;
}
/* Insert; return 1 if new. */
static int insert(const Pos *p) {
    uint64_t k[KEYW];
    pack(p, k);
    if (2 * (nkeys + 1) > nslots) grow();
    uint64_t j = hash(k) & (nslots - 1);
    while (slots[j]) {
        if (!memcmp(keys + (uint64_t)(slots[j] - 1) * KEYW, k, sizeof k)) return 0;
        j = (j + 1) & (nslots - 1);
    }
    if (nkeys == keycap) {
        keycap = keycap ? keycap * 2 : 1 << 16;
        keys = realloc(keys, keycap * KEYW * 8);
        if (!keys) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memcpy(keys + nkeys * KEYW, k, sizeof k);
    slots[j] = (uint32_t)(++nkeys);
    return 1;
}
static void reset(void) {
    nkeys = 0;
    if (slots) memset(slots, 0, nslots * sizeof *slots);
}

/* ---- moves ---- */
typedef struct { uint8_t card, from, to; } Move; /* to: card, BASE, CELL or HOME */

static void do_move(Pos *p, Move m) {
    int c = m.card, f = p->under[c];
    if (f < NCARD) p->covered[f] = 0;
    else if (f == CELL) p->cells--;
    else if (f == BASE) p->bases--;
    if (m.to < NCARD) p->covered[m.to] = 1;
    else if (m.to == CELL) p->cells++;
    else if (m.to == BASE) p->bases++;
    else { p->home[st(c)]++; p->nhome++; }
    p->under[c] = m.to;
}
static void undo_move(Pos *p, Move m) {
    Move back = {m.card, m.to, m.from};
    if (m.to == HOME) { p->home[st(m.card)]--; p->nhome--; p->under[m.card] = m.from;
        if (m.from < NCARD) p->covered[m.from] = 1;
        else if (m.from == CELL) p->cells++;
        else if (m.from == BASE) p->bases++;
        return; }
    do_move(p, back);
}

/* Moves from p, best first by a fixed order of kinds. */
static int gen(const Pos *p, Move *out) {
    int n = 0, free_cols = ncols - p->bases;
    int exposed[NCARD], ne = 0;
    for (int c = 0; c < NCARD; c++)
        if (present[c] && p->under[c] != HOME && !p->covered[c]) exposed[ne++] = c;
    /* 1. to the foundation */
    for (int i = 0; i < ne; i++) {
        int c = exposed[i];
        if (p->home[st(c)] == rk(c) - 1) out[n++] = (Move){c, p->under[c], HOME};
    }
    /* 2. onto a card in a column, from a column then from a cell */
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < ne; i++) {
            int c = exposed[i];
            if ((p->under[c] == CELL) != pass) continue;
            for (int j = 0; j < ne; j++) {
                int t = exposed[j];
                if (t != c && p->under[t] != CELL && stacks(c, t))
                    out[n++] = (Move){c, p->under[c], t};
            }
        }
    /* 3. to an empty column (a column's only card stays put) */
    if (free_cols > 0)
        for (int i = 0; i < ne; i++) {
            int c = exposed[i];
            if (p->under[c] != BASE) out[n++] = (Move){c, p->under[c], BASE};
        }
    /* 4. from a column to a free cell */
    if (p->cells < ncells)
        for (int i = 0; i < ne; i++) {
            int c = exposed[i];
            if (p->under[c] != CELL) out[n++] = (Move){c, p->under[c], CELL};
        }
    return n;
}

/* A score for ordering the moves: lower looks closer to a win. Cards not home, cards in
   cells, cards lying above a lower card, and cards above the next card of each suit. The
   score only changes the order of the search, never its verdict. */
static int score(const Pos *p) {
    int sc = 3 * (deck_size - p->nhome) + 2 * p->cells - 2 * (ncols - p->bases);
    int height[NCARD + 1], depth[NCARD], basecard[NCARD];
    for (int c = 0; c < NCARD; c++) height[c] = 0;
    for (int c = 0; c < NCARD; c++) {
        if (!present[c] || p->under[c] >= BASE) continue;
        int d = 0, x = c, lowest = 99;
        while (p->under[x] < NCARD) { x = p->under[x]; d++; if (rk(x) < lowest) lowest = rk(x); }
        depth[c] = d; basecard[c] = x;
        if (rk(c) > lowest) sc += 2;
    }
    for (int c = 0; c < NCARD; c++)
        if (present[c] && p->under[c] == BASE) { depth[c] = 0; basecard[c] = c; }
    for (int c = 0; c < NCARD; c++)
        if (present[c] && p->under[c] <= BASE && !p->covered[c]) height[basecard[c]] = depth[c] + 1;
    for (int s = 0; s < 4; s++) {
        int c = 4 * p->home[s] + s; /* the next card of suit s */
        if (p->home[s] < nranks && present[c] && p->under[c] <= BASE)
            sc += 2 * (height[basecard[c]] - depth[c] - 1);
    }
    return sc;
}

/* With a nonzero seed, each score gets a random amount from 0 to 7 added, so that a
   restart tries the moves in another order. */
static uint64_t rng;
static int noise(void) {
    if (!rng) return 0;
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (int)(rng % 8);
}

static void order(Pos *p, Move *m, int n) {
    int key[1024];
    for (int i = 0; i < n; i++) {
        do_move(p, m[i]);
        key[i] = m[i].to == HOME ? -1000 : score(p) + noise();
        undo_move(p, m[i]);
    }
    for (int i = 1; i < n; i++) { /* insertion sort, stable */
        Move x = m[i]; int k = key[i], j = i - 1;
        while (j >= 0 && key[j] > k) { m[j + 1] = m[j]; key[j + 1] = key[j]; j--; }
        m[j + 1] = x; key[j + 1] = k;
    }
}

/* ---- depth-first search with an explicit stack ---- */
typedef struct { uint64_t first; int count, next, nauto; Move made; } Frame;
static Move *mv; static uint64_t mvcap;
static uint8_t moved[NCARD]; /* cards moved on the way to some stored position */
static Frame *fr; static uint64_t frcap;

/* Optional autoplay ("auto" on the command line): after each move, play home every card
   that can go home and on which only cards already home, or aces, could be placed. The
   rule is the one Solver A uses, written again here. The cards played go on a stack of
   their own, so that a step back undoes them with the move. */
static int use_auto;
static Move *au; static uint64_t aucap, nau;

static int safe_home(const Pos *p, int c) {
    if (p->home[st(c)] != rk(c) - 1) return 0;
    for (int d = 0; d < NCARD; d++)
        if (present[d] && rk(d) == rk(c) - 1 && red(d) != red(c) && rk(d) != 1 &&
            p->home[st(d)] < rk(d))
            return 0;
    return 1;
}
static int settle(Pos *p) {
    int count = 0, changed = use_auto;
    while (changed) {
        changed = 0;
        for (int c = 0; c < NCARD; c++)
            if (present[c] && p->under[c] != HOME && !p->covered[c] && safe_home(p, c)) {
                Move a = {c, p->under[c], HOME};
                do_move(p, a);
                if (nau == aucap) {
                    aucap = aucap ? 2 * aucap : 1 << 12;
                    au = realloc(au, aucap * sizeof *au);
                    if (!au) { fprintf(stderr, "out of memory\n"); exit(2); }
                }
                au[nau++] = a;
                count++;
                changed = 1;
            }
    }
    return count;
}
static void unsettle(Pos *p, int k) {
    while (k-- > 0) undo_move(p, au[--nau]);
}

static void need(uint64_t moves, uint64_t frames) {
    if (moves > mvcap) { while (moves > mvcap) mvcap = mvcap ? mvcap * 2 : 1 << 16;
        mv = realloc(mv, mvcap * sizeof *mv); }
    if (frames > frcap) { while (frames > frcap) frcap = frcap ? frcap * 2 : 1 << 12;
        fr = realloc(fr, frcap * sizeof *fr); }
    if (!mv || !fr) { fprintf(stderr, "out of memory\n"); exit(2); }
}

static const char *dest(int to, char *buf) {
    if (to == HOME) return "h";
    if (to == CELL) return "c";
    if (to == BASE) return "e";
    sprintf(buf, "%d", to);
    return buf;
}

static void solve(Pos *p, uint64_t budget, int want_fence) {
    reset();
    nau = 0;
    memset(moved, 0, sizeof moved);
    int root_auto = settle(p);
    for (uint64_t i = 0; i < nau; i++) moved[au[i].card] = 1;
    insert(p);
    int most = p->nhome;
    char buf[8];
    if (p->nhome == deck_size) {
        printf("won 1");
        for (uint64_t i = 0; i < nau; i++) printf(" %d>h", au[i].card);
        printf("\n");
        return;
    }
    uint64_t depth = 0, top = 0;
    need(1024, 1);
    int n = gen(p, mv);
    order(p, mv, n);
    fr[0] = (Frame){0, n, 0, root_auto, {0, 0, 0}};
    top = n;
    for (;;) {
        Frame *f = &fr[depth];
        if (f->next == f->count) {
            if (depth == 0) break;
            top = f->first;
            unsettle(p, f->nauto);
            undo_move(p, f->made);
            depth--;
            continue;
        }
        Move m = mv[f->first + f->next++];
        do_move(p, m);
        int k = settle(p);
        if (!insert(p)) { unsettle(p, k); undo_move(p, m); continue; }
        moved[m.card] = 1;
        for (uint64_t i = nau - k; i < nau; i++) moved[au[i].card] = 1;
        if (p->nhome > most) most = p->nhome;
        if (p->nhome == deck_size) {
            printf("won %llu", (unsigned long long)nkeys);
            uint64_t a = 0;
            for (uint64_t d = 0; d <= depth; d++) {
                if (d) printf(" %d>%s", fr[d].made.card, dest(fr[d].made.to, buf));
                for (int i = 0; i < fr[d].nauto; i++) printf(" %d>h", au[a++].card);
            }
            printf(" %d>%s", m.card, dest(m.to, buf));
            while (a < nau) printf(" %d>h", au[a++].card);
            printf("\n");
            return;
        }
        if (budget && nkeys >= budget) { printf("unknown %llu\n", (unsigned long long)nkeys); return; }
        need(top + 1024, depth + 2);
        n = gen(p, mv + top);
        order(p, mv + top, n);
        depth++;
        fr[depth] = (Frame){top, n, 0, k, m};
        top += n;
    }
    printf("lost %llu %d", (unsigned long long)nkeys, most);
    for (int c = 0; c < NCARD; c++)
        if (present[c] && !moved[c]) printf(" %d", c);
    printf("\n");
    if (want_fence) {
        for (uint64_t i = 0; i < nkeys; i++) {
            const uint64_t *k = keys + i * KEYW;
            for (int c = 0; c < NCARD; c++) {
                int bit = 6 * c;
                uint64_t v = k[bit / 64] >> (bit % 64);
                if (bit % 64 > 58) v |= k[bit / 64 + 1] << (64 - bit % 64);
                printf(c ? " %d" : "%d", (int)(v & 63));
            }
            printf("\n");
        }
        printf("end\n");
    }
}

/* ---- one foundation at a time ---- */
static void unpack(const uint64_t *k, Pos *p) {
    memset(p, 0, sizeof *p);
    for (int c = 0; c < NCARD; c++) {
        int bit = 6 * c;
        uint64_t v = k[bit / 64] >> (bit % 64);
        if (bit % 64 > 58) v |= k[bit / 64 + 1] << (64 - bit % 64);
        p->under[c] = (uint8_t)(v & 63);
    }
    for (int c = 0; c < NCARD; c++) {
        int u = p->under[c];
        if (!present[c]) continue;
        if (u < NCARD) p->covered[u] = 1;
        else if (u == CELL) p->cells++;
        else if (u == BASE) p->bases++;
        else if (u == HOME) { p->home[st(c)]++; p->nhome++; }
    }
}
static int foundation_id(const Pos *p) {
    return p->home[0] + 14 * (p->home[1] + 14 * (p->home[2] + 14 * p->home[3]));
}
typedef struct { uint64_t *k; uint64_t n, cap; } List;
static List waiting[14 * 14 * 14 * 14];

static void wait_for(int id, const Pos *p) {
    List *l = &waiting[id];
    if (l->n == l->cap) {
        l->cap = l->cap ? 2 * l->cap : 64;
        l->k = realloc(l->k, l->cap * KEYW * 8);
        if (!l->k) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    pack(p, l->k + l->n * KEYW);
    l->n++;
}

static void solve_layers(Pos *p) {
    use_auto = 1;
    nau = 0;
    memset(moved, 0, sizeof moved);
    settle(p);
    for (uint64_t i = 0; i < nau; i++) moved[au[i].card] = 1;
    nau = 0;
    int most = p->nhome, won = 0;
    uint64_t total = 0, nlog = 0;
    static int logid[14 * 14 * 14 * 14];
    static uint64_t logn[14 * 14 * 14 * 14];
    if (p->nhome == deck_size) won = 1;
    else wait_for(foundation_id(p), p);
    Move moves[1024];
    while (!won) {
        int cur = -1, fewest = 99;
        for (int id = 0; id < 14 * 14 * 14 * 14; id++)
            if (waiting[id].n) {
                int h = id % 14 + id / 14 % 14 + id / 196 % 14 + id / 2744;
                if (h < fewest) { fewest = h; cur = id; }
            }
        if (cur < 0) break;
        reset();
        Pos q;
        for (uint64_t i = 0; i < waiting[cur].n; i++) {
            unpack(waiting[cur].k + i * KEYW, &q);
            insert(&q);
        }
        free(waiting[cur].k);
        memset(&waiting[cur], 0, sizeof waiting[cur]);
        if (fewest > most) most = fewest;
        /* the stored positions, in the order stored, are the queue */
        for (uint64_t i = 0; i < nkeys && !won; i++) {
            unpack(keys + i * KEYW, &q);
            int n = gen(&q, moves);
            for (int j = 0; j < n && !won; j++) {
                do_move(&q, moves[j]);
                int k = settle(&q);
                if (q.nhome == deck_size) won = 1;
                else if (foundation_id(&q) != cur) {
                    wait_for(foundation_id(&q), &q);
                    moved[moves[j].card] = 1;
                    for (uint64_t a = nau - k; a < nau; a++) moved[au[a].card] = 1;
                } else if (insert(&q)) {
                    moved[moves[j].card] = 1;
                    for (uint64_t a = nau - k; a < nau; a++) moved[au[a].card] = 1;
                }
                unsettle(&q, k);
                undo_move(&q, moves[j]);
            }
        }
        total += nkeys;
        logid[nlog] = cur; logn[nlog++] = nkeys;
        if (getenv("SOLVER_B_PROGRESS"))
            fprintf(stderr, "layer %d: %llu (total %llu)\n", cur, (unsigned long long)nkeys,
                    (unsigned long long)total);
    }
    if (won) printf("won %llu\n", (unsigned long long)total);
    else {
        printf("lost %llu %d", (unsigned long long)total, most);
        for (int c = 0; c < NCARD; c++)
            if (present[c] && !moved[c]) printf(" %d", c);
        printf("\n");
    }
    for (uint64_t i = 0; i < nlog; i++)
        printf("layer %d %d %d %d %llu\n", logid[i] % 14, logid[i] / 14 % 14,
               logid[i] / 196 % 14, logid[i] / 2744, (unsigned long long)logn[i]);
    printf("end\n");
    for (int id = 0; id < 14 * 14 * 14 * 14; id++) {
        free(waiting[id].k);
        memset(&waiting[id], 0, sizeof waiting[id]);
    }
    reset();
}

/* ---- setting up a problem ---- */
static void setup(Pos *p, int cols[][NCARD], int *len) {
    memset(p, 0, sizeof *p);
    memset(p->under, ABSENT, sizeof p->under);
    for (int i = 0; i < ncols; i++)
        for (int j = 0; j < len[i]; j++) {
            int c = cols[i][j];
            p->under[c] = j ? cols[i][j - 1] : BASE;
            if (j) p->covered[cols[i][j - 1]] = 1;
            else p->bases++;
        }
}

/* Microsoft deal: the C runtime's rand() seeded with the deal number. */
static void ms_layout(long deal, int cols[][NCARD], int *len) {
    uint32_t seed = (uint32_t)deal;
    int deck[52], n = 52;
    for (int i = 0; i < 52; i++) deck[i] = i;
    for (int i = 0; i < 8; i++) len[i] = 0;
    for (int k = 0; k < 52; k++) {
        seed = (seed * 214013u + 2531011u) & 0x7fffffffu;
        int j = (int)((seed >> 16) % (uint32_t)n);
        cols[k % 8][len[k % 8]++] = deck[j];
        deck[j] = deck[--n];
    }
}

int main(int argc, char **argv) {
    uint64_t budget = argc > 1 ? strtoull(argv[1], 0, 10) : 0;
    uint64_t seed = argc > 2 ? strtoull(argv[2], 0, 10) : 0;
    int want_fence = 0, by_layers = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "fence")) want_fence = 1;
        else if (!strcmp(argv[i], "auto")) use_auto = 1;
        else if (!strcmp(argv[i], "layers")) by_layers = 1;
    }
    char line[4096];
    static int cols[16][NCARD];
    int len[16];
    Pos p;
    while (fgets(line, sizeof line, stdin)) {
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        memset(present, 0, sizeof present);
        if (!strcmp(tok, "ms")) {
            long deal = atol(strtok(0, " \t\n"));
            ncells = atoi(strtok(0, " \t\n"));
            nranks = 13; nsuits = 4; ncols = 8;
            ms_layout(deal, cols, len);
        } else if (!strcmp(tok, "deck")) {
            nranks = atoi(strtok(0, " \t\n"));
            nsuits = atoi(strtok(0, " \t\n"));
            ncols = atoi(strtok(0, " \t\n"));
            ncells = atoi(strtok(0, " \t\n"));
            int i = 0;
            len[0] = 0;
            while ((tok = strtok(0, " \t\n"))) {
                if (!strcmp(tok, "|")) { len[++i] = 0; continue; }
                cols[i][len[i]++] = atoi(tok);
            }
            if (i + 1 != ncols) { printf("error columns\n"); fflush(stdout); continue; }
        } else { printf("error input\n"); fflush(stdout); continue; }
        deck_size = 0;
        for (int i = 0; i < ncols; i++)
            for (int j = 0; j < len[i]; j++) { present[cols[i][j]] = 1; deck_size++; }
        setup(&p, cols, len);
        rng = seed ? seed * 0x9E3779B97F4A7C15ULL : 0;
        if (by_layers) solve_layers(&p);
        else solve(&p, budget, want_fence);
        fflush(stdout);
    }
    return 0;
}
