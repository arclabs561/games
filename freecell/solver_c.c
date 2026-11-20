/* Solver C: an exhaustive FreeCell search in layers, one layer per foundation.

   Cards never leave a foundation, so the four foundation heights only grow. Group the
   positions by their foundations, a "layer". A move either stays in its layer or goes up to a
   layer with more cards home. So the layers can be searched one at a time, in order of cards
   home: when a layer is searched, every position that can enter it from below is already
   known. Only one layer's positions are held in memory; positions that enter a higher layer
   wait in a file until it is searched, and a layer is forgotten once it is done. The deal is
   won when the layer with every card home is reached, and lost when no layer is left.

   Safe cards are played home after every move, by the rule of rules.py (a card goes home when
   it can and every card that could sit on it, one rank lower of the other color, is home or
   is an ace). With the word "horne", the rule is Horne's instead, which the solvers of
   Fish's reports use: a card of rank r goes home when it can and the foundations of the
   other color are at r - 2 or more and the other foundation of its color at r - 3 or more. A position is the multiset of columns, the set of free cells and the
   foundations; the count of positions is comparable with Solver B run with "auto".

   A position is written in few bits, relative to the deal. Each column of the deal keeps its
   bottom cards in place until one of them moves, so a column on the table is the longest
   start of some dealt column that it still has, then a run above it, each card one rank
   lower and of the other color than the card below it. Such a card needs one bit (which of
   the two suits), and one more bit says whether the run goes on. A column whose bottom card
   is not the bottom card of its dealt column is all run, and gives its bottom card in full.
   The code:
     for each dealt column: 3 bits, how many of its cards are still in place (0 to 7);
       if some are, the run above them: per card a 1 bit and the suit bit, then a 0 bit;
     4 bits, the number of other columns; each, in order of its bottom card: 6 bits for that
       card, then its run as above;
     3 bits, the number of cards in free cells; each, in increasing order, 6 bits.
   The foundations are those of the layer and are not written. Codes of at most 128 bits are
   kept in a table of 16-byte keys, longer ones (up to 256 bits) in a second table.

   Input, one problem per line on stdin (as Solver B):
     ms <deal> <cells>
     deck <ranks> <suits> <columns> <cells> <card> ... | ...    (columns of at most 7 cards)
   Output, one line per problem:
     won <positions> <most cards home>
     lost <positions> <most cards home> <cards that never move...>
     unknown <positions>
   then, with the word "layers" on the command line, one line per layer searched:
     layer <home of clubs> <diamonds> <hearts> <spades> <positions>
   and a line "end".
   Arguments: a folder for the waiting files (default /tmp), the most positions in one layer
   (0 for no limit), the most bytes of waiting positions held in memory before they are
   written to files (default 1 GB), and optionally "layers", "horne" and "win=<budget>"
   (the hunt for a line that wins, below).

   A win is reported without its line of moves; Solver B or Solver A finds and replays
   lines. Solver C shares no code with either.
*/
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MAXCOL 16
#define MAXH 40

static int nranks, nsuits, ncols, ncells, decksize;
static int inplay[52];
static uint8_t dealt[MAXCOL][8];
static int dealtlen[MAXCOL];
static int bottom_of[52]; /* dealt column whose bottom card this is, or -1 */

static inline int rank_of(int c) { return c / 4 + 1; }
static inline int suit_of(int c) { return c % 4; }
static inline int is_red(int c) { int s = c % 4; return s == 1 || s == 2; }
static inline int fits(int c, int onto) {
    return rank_of(onto) == rank_of(c) + 1 && is_red(onto) != is_red(c);
}
/* The suit bit of card c sitting on a card of the other color. */
static inline int suit_bit(int c) { return suit_of(c) == 2 || suit_of(c) == 3; }
/* The card one rank below `onto`, of the other color, with suit bit b. */
static inline int card_on(int onto, int b) {
    int s = is_red(onto) ? (b ? 3 : 0) : (b ? 2 : 1);
    return 4 * (rank_of(onto) - 2) + s;
}

typedef struct {
    uint8_t col[MAXCOL][MAXH];
    uint8_t len[MAXCOL];
    uint8_t cell[8];
    uint8_t ncell;
    uint8_t home[4];
} Pos;

/* ---- bit codes ---- */
typedef struct { uint64_t w[4]; int n; } Code;

static inline void put(Code *k, uint64_t v, int bits) {
    for (int i = 0; i < bits; i++, k->n++)
        if (v >> i & 1) k->w[k->n >> 6] |= 1ULL << (k->n & 63);
}
typedef struct { const uint64_t *w; int n; } Reader;
static inline uint64_t get(Reader *r, int bits) {
    uint64_t v = 0;
    for (int i = 0; i < bits; i++, r->n++) v |= (r->w[r->n >> 6] >> (r->n & 63) & 1) << i;
    return v;
}

static void put_run(Code *k, const uint8_t *cards, int from, int to) {
    for (int i = from; i < to; i++) { put(k, 1, 1); put(k, suit_bit(cards[i]), 1); }
    put(k, 0, 1);
}

static void encode(const Pos *p, Code *k) {
    memset(k, 0, sizeof *k);
    int slot[MAXCOL], kept[MAXCOL];
    for (int j = 0; j < ncols; j++) slot[j] = -1;
    int others[MAXCOL], nothers = 0;
    for (int i = 0; i < ncols; i++) {
        if (!p->len[i]) continue;
        int j = bottom_of[p->col[i][0]];
        if (j < 0) { others[nothers++] = i; continue; }
        int m = 0;
        while (m < p->len[i] && m < dealtlen[j] && p->col[i][m] == dealt[j][m]) m++;
        slot[j] = i; kept[j] = m;
    }
    for (int j = 0; j < ncols; j++) {
        if (slot[j] < 0) { put(k, 0, 3); continue; }
        int i = slot[j];
        put(k, kept[j], 3);
        put_run(k, p->col[i], kept[j], p->len[i]);
    }
    for (int a = 1; a < nothers; a++) { /* by bottom card */
        int x = others[a], b = a - 1;
        while (b >= 0 && p->col[others[b]][0] > p->col[x][0]) { others[b + 1] = others[b]; b--; }
        others[b + 1] = x;
    }
    put(k, nothers, 4);
    for (int a = 0; a < nothers; a++) {
        int i = others[a];
        put(k, p->col[i][0], 6);
        put_run(k, p->col[i], 1, p->len[i]);
    }
    uint8_t cs[8];
    memcpy(cs, p->cell, p->ncell);
    for (int a = 1; a < p->ncell; a++) {
        uint8_t x = cs[a]; int b = a - 1;
        while (b >= 0 && cs[b] > x) { cs[b + 1] = cs[b]; b--; }
        cs[b + 1] = x;
    }
    put(k, p->ncell, 3);
    for (int a = 0; a < p->ncell; a++) put(k, cs[a], 6);
}

static void get_run(Reader *r, Pos *p, int i) {
    while (get(r, 1)) {
        int top = p->col[i][p->len[i] - 1];
        p->col[i][p->len[i]++] = (uint8_t)card_on(top, (int)get(r, 1));
    }
}

static void decode(const uint64_t *w, const uint8_t *home, Pos *p) {
    Reader r = {w, 0};
    memset(p->len, 0, sizeof p->len);
    int i = 0;
    for (int j = 0; j < ncols; j++) {
        int m = (int)get(&r, 3);
        if (!m) continue;
        memcpy(p->col[i], dealt[j], m);
        p->len[i] = m;
        get_run(&r, p, i);
        i++;
    }
    int nothers = (int)get(&r, 4);
    for (int a = 0; a < nothers; a++, i++) {
        p->col[i][0] = (uint8_t)get(&r, 6);
        p->len[i] = 1;
        get_run(&r, p, i);
    }
    p->ncell = (uint8_t)get(&r, 3);
    for (int a = 0; a < p->ncell; a++) p->cell[a] = (uint8_t)get(&r, 6);
    memcpy(p->home, home, 4);
}

/* ---- a set of codes of one width, with the codes in order of insertion ---- */
typedef struct {
    int words;          /* 2 or 4 */
    uint64_t *keys;     /* reserved address space, filled in order */
    uint64_t n, reserved;
    uint32_t *slots;    /* index + 1, 0 for empty */
    uint64_t nslots;
} Table;

static uint64_t mix(const uint64_t *k, int words) {
    uint64_t h = 0x243F6A8885A308D3ULL;
    for (int i = 0; i < words; i++) {
        h = (h ^ k[i]) * 0x9FB21C651E98DF25ULL;
        h ^= h >> 29;
    }
    return h * 0xD6E8FEB86659FD93ULL;
}

static void table_init(Table *t, int words, uint64_t most) {
    t->words = words;
    t->reserved = most;
    t->keys = mmap(0, most * words * 8, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (t->keys == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(2); }
    t->n = 0;
    t->nslots = 1 << 16;
    t->slots = calloc(t->nslots, 4);
}
static void table_clear(Table *t) {
    madvise(t->keys, t->reserved * t->words * 8, MADV_DONTNEED);
    t->n = 0;
    free(t->slots);
    t->nslots = 1 << 16;
    t->slots = calloc(t->nslots, 4);
}
static void table_grow(Table *t) {
    uint64_t n2 = t->nslots * 2;
    uint32_t *s2 = calloc(n2, 4);
    if (!s2) { fprintf(stderr, "out of memory\n"); exit(2); }
    for (uint64_t i = 0; i < t->n; i++) {
        uint64_t j = mix(t->keys + i * t->words, t->words) & (n2 - 1);
        while (s2[j]) j = (j + 1) & (n2 - 1);
        s2[j] = (uint32_t)(i + 1);
    }
    free(t->slots);
    t->slots = s2;
    t->nslots = n2;
}
/* 1 if new, 0 if present, -1 if full */
static int table_add(Table *t, const uint64_t *k) {
    if (4 * (t->n + 1) > 3 * t->nslots) table_grow(t);
    uint64_t j = mix(k, t->words) & (t->nslots - 1);
    while (t->slots[j]) {
        if (!memcmp(t->keys + (uint64_t)(t->slots[j] - 1) * t->words, k, t->words * 8)) return 0;
        j = (j + 1) & (t->nslots - 1);
    }
    if (t->n == t->reserved || t->n == 0xFFFFFFFEULL) return -1;
    memcpy(t->keys + t->n * t->words, k, t->words * 8);
    t->slots[j] = (uint32_t)(++t->n);
    return 1;
}

/* ---- waiting positions of higher layers ---- */
#define NLAYER (14 * 14 * 14 * 14)
static int layer_id(const uint8_t *h) { return h[0] + 14 * (h[1] + 14 * (h[2] + 14 * h[3])); }
static void layer_home(int id, uint8_t *h) { for (int s = 0; s < 4; s++) { h[s] = id % 14; id /= 14; } }

/* A layer is split further, into **sublayers**. In each dealt column, the cards still in
   their dealt places end, at the top, in a run where each card fits on the one below it (it
   may be a single card). Where that run starts, K, never rises: when the top card of the
   kept cards leaves, K stays or falls, and a card can only come back to its dealt place by
   fitting on the card below it, which lengthens the run. So a position's sublayer, its
   foundations and the eight K, can only change to one with more cards home or a lower K,
   and the number of cards home minus the sum of the K grows at each change of sublayer.
   Sublayers are searched in order of that number; each is complete when its turn comes.
   Without the word "fine", the K are left out and a sublayer is a whole layer. */
static int fine;
static char folder[512] = "/tmp";
static uint64_t waiting_bytes, memory_limit = 1ULL << 30;
static uint8_t pending[NLAYER]; /* used by the hunt for a win */

typedef struct {
    uint64_t key, *buf16, *buf32, n16, n32, cap16, cap32;
    int disk16, disk32, queued;
} Wait;
static Wait *waits; static uint64_t nwaits, capwaits;
static uint64_t *wmap, wmapsize; /* index + 1 */
#define NPRIO 160
static uint64_t *bucket[NPRIO], nbucket[NPRIO], capbucket[NPRIO];

static int sub_prio(uint64_t key) {
    int lid = (int)(key >> 24), k = 0;
    uint8_t h[4];
    layer_home(lid, h);
    for (int j = 0; j < 8; j++) k += (int)(key >> (3 * j) & 7);
    return h[0] + h[1] + h[2] + h[3] - k + 64;
}
static uint64_t sub_key(const Pos *p) {
    uint64_t key = (uint64_t)layer_id(p->home) << 24;
    if (!fine) return key;
    for (int i = 0; i < ncols; i++) {
        if (!p->len[i]) continue;
        int j = bottom_of[p->col[i][0]];
        if (j < 0) continue;
        int m = 0;
        while (m < p->len[i] && m < dealtlen[j] && p->col[i][m] == dealt[j][m]) m++;
        int K = m - 1;
        while (K >= 1 && fits(dealt[j][K], dealt[j][K - 1])) K--;
        key |= (uint64_t)K << (3 * j);
    }
    return key;
}
static uint64_t mixkey(uint64_t k) { k *= 0x9E3779B97F4A7C15ULL; return k ^ (k >> 31); }
static Wait *find_wait(uint64_t key) {
    if (2 * (nwaits + 1) > wmapsize) {
        uint64_t n2 = wmapsize ? 2 * wmapsize : 1 << 12;
        uint64_t *m2 = calloc(n2, 8);
        for (uint64_t i = 0; i < nwaits; i++) {
            uint64_t j = mixkey(waits[i].key) & (n2 - 1);
            while (m2[j]) j = (j + 1) & (n2 - 1);
            m2[j] = i + 1;
        }
        free(wmap); wmap = m2; wmapsize = n2;
    }
    uint64_t j = mixkey(key) & (wmapsize - 1);
    while (wmap[j]) {
        if (waits[wmap[j] - 1].key == key) return &waits[wmap[j] - 1];
        j = (j + 1) & (wmapsize - 1);
    }
    if (nwaits == capwaits) {
        capwaits = capwaits ? 2 * capwaits : 1024;
        waits = realloc(waits, capwaits * sizeof *waits);
        if (!waits) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memset(&waits[nwaits], 0, sizeof *waits);
    waits[nwaits].key = key;
    wmap[j] = ++nwaits;
    return &waits[nwaits - 1];
}
static void wait_path(char *buf, uint64_t key, int words) {
    snprintf(buf, 600, "%s/fc-c-%d-%llx-%d.bin", folder, (int)getpid(), (unsigned long long)key,
             words);
}
static void spill(Wait *q, int words) {
    uint64_t **buf = words == 2 ? &q->buf16 : &q->buf32;
    uint64_t *n = words == 2 ? &q->n16 : &q->n32, *cap = words == 2 ? &q->cap16 : &q->cap32;
    if (!*n) return;
    char path[600];
    wait_path(path, q->key, words);
    FILE *f = fopen(path, "ab");
    if (!f || fwrite(*buf, 8 * words, *n, f) != *n) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fclose(f);
    if (words == 2) q->disk16 = 1; else q->disk32 = 1;
    waiting_bytes -= *n * 8 * words;
    free(*buf); *buf = 0; *n = 0; *cap = 0;
}
/* Write the larger waiting lists to files until half the limit is left in memory. */
static void flush_some(void) {
    for (uint64_t least = 1 << 20; waiting_bytes > memory_limit / 2 && least >= 1; least /= 16)
        for (uint64_t i = 0; i < nwaits && waiting_bytes > memory_limit / 2; i++) {
            if (waits[i].n16 * 16 >= least) spill(&waits[i], 2);
            if (waits[i].n32 * 32 >= least) spill(&waits[i], 4);
        }
}
static void wait_add(uint64_t key, const Code *k) {
    int words = k->n <= 128 ? 2 : 4;
    Wait *q = find_wait(key);
    uint64_t **buf = words == 2 ? &q->buf16 : &q->buf32;
    uint64_t *n = words == 2 ? &q->n16 : &q->n32, *cap = words == 2 ? &q->cap16 : &q->cap32;
    if (*n == *cap) {
        *cap = *cap ? 2 * *cap : 16;
        *buf = realloc(*buf, *cap * 8 * words);
        if (!*buf) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memcpy(*buf + *n * words, k->w, 8 * words);
    (*n)++;
    waiting_bytes += 8 * words;
    if (!q->queued) {
        int pr = sub_prio(key);
        q->queued = 1;
        if (nbucket[pr] == capbucket[pr]) {
            capbucket[pr] = capbucket[pr] ? 2 * capbucket[pr] : 64;
            bucket[pr] = realloc(bucket[pr], capbucket[pr] * 8);
        }
        bucket[pr][nbucket[pr]++] = (uint64_t)(q - waits);
    }
    if (waiting_bytes > memory_limit) flush_some();
}
static void wait_clear_all(void) {
    char path[600];
    for (uint64_t i = 0; i < nwaits; i++) {
        if (waits[i].disk16) { wait_path(path, waits[i].key, 2); unlink(path); }
        if (waits[i].disk32) { wait_path(path, waits[i].key, 4); unlink(path); }
        free(waits[i].buf16); free(waits[i].buf32);
    }
    nwaits = 0;
    if (wmap) memset(wmap, 0, wmapsize * 8);
    for (int p = 0; p < NPRIO; p++) nbucket[p] = 0;
    waiting_bytes = 0;
    memset(pending, 0, sizeof pending);
}

/* ---- moves ---- */
static int horne;
static int suit_in_play[4];
static int safe(const Pos *p, int c) {
    if (p->home[suit_of(c)] != rank_of(c) - 1) return 0;
    int r = rank_of(c);
    if (r <= 2) return 1;
    if (horne) { /* Horne's rule: other color at least r - 2, same color at least r - 3 */
        for (int s = 0; s < 4; s++) {
            if (!suit_in_play[s] || s == suit_of(c)) continue;
            int need = is_red(s) != is_red(c) ? r - 2 : r - 3;
            if (p->home[s] < need) return 0;
        }
        return 1;
    }
    for (int s = 0; s < 4; s++) {
        int d = 4 * (r - 2) + s;
        if (!inplay[d] || is_red(d) == is_red(c)) continue;
        if (p->home[s] < r - 1) return 0;
    }
    return 1;
}
/* Play safe cards home until none is left; the cards played go to `played`. */
static int autoplay(Pos *p, uint8_t *played) {
    int n = 0;
    for (int again = 1; again;) {
        again = 0;
        for (int i = 0; i < ncols; i++)
            while (p->len[i] && safe(p, p->col[i][p->len[i] - 1])) {
                int c = p->col[i][--p->len[i]];
                p->home[suit_of(c)]++;
                played[n++] = (uint8_t)c;
                again = 1;
            }
        for (int a = 0; a < p->ncell; a++)
            if (safe(p, p->cell[a])) {
                int c = p->cell[a];
                p->home[suit_of(c)]++;
                played[n++] = (uint8_t)c;
                p->cell[a] = p->cell[--p->ncell];
                a--;
                again = 1;
            }
    }
    return n;
}
static int home_total(const uint8_t *h) { return h[0] + h[1] + h[2] + h[3]; }

enum { TO_HOME = -1, TO_CELL = -2, TO_EMPTY = -3 };
/* Called with each position q one move after p, safe cards played: the card moved, where
   it went (a card it went onto, or TO_HOME, TO_CELL, TO_EMPTY) and the cards played home
   after it. A nonzero return stops the listing and is returned. */
typedef int (*Visit)(void *ctx, Pos *q, int card, int dest, const uint8_t *played, int nplayed);

static int successors(const Pos *p, Visit f, void *ctx) {
    Pos q;
    uint8_t played[52];
    int empty = -1, r;
    for (int i = 0; i < ncols; i++) if (!p->len[i]) { empty = i; break; }
    for (int src = 0; src < ncols + p->ncell; src++) {
        int c, fromcol = src < ncols;
        if (fromcol) { if (!p->len[src]) continue; c = p->col[src][p->len[src] - 1]; }
        else c = p->cell[src - ncols];
#define LIFT() do { q = *p; if (fromcol) q.len[src]--; \
                    else { q.cell[src - ncols] = q.cell[--q.ncell]; } } while (0)
#define SEND(dest) do { int k_ = autoplay(&q, played); \
                        if ((r = f(ctx, &q, c, (dest), played, k_))) return r; } while (0)
        if (p->home[suit_of(c)] == rank_of(c) - 1) {
            LIFT(); q.home[suit_of(c)]++; SEND(TO_HOME);
        }
        for (int t = 0; t < ncols; t++) {
            if (t == src && fromcol) continue;
            if (!p->len[t] || !fits(c, p->col[t][p->len[t] - 1])) continue;
            int onto = p->col[t][p->len[t] - 1];
            LIFT(); q.col[t][q.len[t]++] = (uint8_t)c; SEND(onto);
        }
        if (empty >= 0 && !(fromcol && p->len[src] == 1)) {
            LIFT(); q.col[empty][0] = (uint8_t)c; q.len[empty] = 1; SEND(TO_EMPTY);
        }
        if (fromcol && p->ncell < ncells) {
            LIFT(); q.cell[q.ncell++] = (uint8_t)c; SEND(TO_CELL);
        }
#undef SEND
#undef LIFT
    }
    return 0;
}

/* ---- deciding a level: layers in order of cards home ---- */
static Table t16, t32;
static uint64_t layer_cap, budget;
static int want_layers, win_mode;


typedef struct { uint64_t cur; int won, full; uint64_t moved; } Decide;

static int decide_visit(void *ctx, Pos *q, int card, int dest, const uint8_t *played, int n) {
    Decide *d = ctx;
    (void)dest;
    d->moved |= 1ULL << card;
    for (int i = 0; i < n; i++) d->moved |= 1ULL << played[i];
    if (home_total(q->home) == decksize) { d->won = 1; return 1; }
    Code k;
    encode(q, &k);
    uint64_t key = sub_key(q);
    if (key != d->cur) { wait_add(key, &k); return 0; }
    int r = k.n <= 128 ? table_add(&t16, k.w) : table_add(&t32, k.w);
    if (r < 0) d->full = 1;
    return 0;
}

static void load_waiting(Wait *q, int *full) {
    char path[600];
    for (int w = 0; w < 2; w++) {
        int words = w ? 4 : 2, disk = w ? q->disk32 : q->disk16;
        Table *t = w ? &t32 : &t16;
        if (disk) {
            wait_path(path, q->key, words);
            FILE *f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
            uint64_t buf[4 * 4096];
            size_t got;
            while ((got = fread(buf, 8 * words, 4096, f)) > 0)
                for (size_t i = 0; i < got; i++)
                    if (table_add(t, buf + i * words) < 0) *full = 1;
            fclose(f);
            unlink(path);
        }
        uint64_t n = w ? q->n32 : q->n16, *buf = w ? q->buf32 : q->buf16;
        for (uint64_t i = 0; i < n; i++)
            if (table_add(t, buf + i * words) < 0) *full = 1;
        waiting_bytes -= n * 8 * words;
        free(buf);
    }
    q->buf16 = q->buf32 = 0;
    q->n16 = q->n32 = q->cap16 = q->cap32 = 0;
    q->disk16 = q->disk32 = 0;
}

static uint64_t layer_count[NLAYER];
static uint8_t layer_seen[NLAYER];

static void decide(Pos *start) {
    uint8_t played[52];
    Decide d = {0, 0, 0, 0};
    uint64_t total = 0, sublayers = 0, largest = 0;
    int most = 0;
    memset(layer_count, 0, sizeof layer_count);
    memset(layer_seen, 0, sizeof layer_seen);
    int n = autoplay(start, played);
    for (int i = 0; i < n; i++) d.moved |= 1ULL << played[i];
    if (home_total(start->home) == decksize) d.won = 1;
    else {
        Code k;
        encode(start, &k);
        wait_add(sub_key(start), &k);
    }
    time_t t0 = time(0), last = t0;
    for (int pr = 0; pr < NPRIO && !d.won && !d.full; pr++)
        for (uint64_t b = 0; b < nbucket[pr] && !d.won && !d.full; b++) {
            Wait *q = &waits[bucket[pr][b]];
            uint64_t cur = q->key;
            int lid = (int)(cur >> 24);
            uint8_t home[4];
            layer_home(lid, home);
            if (home_total(home) > most) most = home_total(home);
            table_clear(&t16); table_clear(&t32);
            d.cur = cur;
            load_waiting(q, &d.full);
            q = 0; /* waits may move while the layer is searched */
            Pos p;
            /* Each table is read in order of insertion; both grow while they are read. */
            uint64_t i16 = 0, i32 = 0;
            while (!d.won && !d.full && (i16 < t16.n || i32 < t32.n)) {
                if (i16 < t16.n) decode(t16.keys + 2 * i16++, home, &p);
                else decode(t32.keys + 4 * i32++, home, &p);
                successors(&p, decide_visit, &d);
                if (layer_cap && t16.n + t32.n > layer_cap) d.full = 1;
            }
            uint64_t count = t16.n + t32.n;
            total += count;
            sublayers++;
            if (count > largest) largest = count;
            layer_count[lid] += count;
            layer_seen[lid] = 1;
            if (getenv("SOLVER_C_PROGRESS") && (count > 1000000 || time(0) - last >= 60)) {
                last = time(0);
                fprintf(stderr, "layer %d %d %d %d: %llu (total %llu, waiting %llu MB, %lds)\n",
                        home[0], home[1], home[2], home[3], (unsigned long long)count,
                        (unsigned long long)total, (unsigned long long)(waiting_bytes >> 20),
                        (long)(time(0) - t0));
            }
        }
    if (d.won) printf("won %llu %d\n", (unsigned long long)total, decksize);
    else if (d.full) printf("unknown %llu\n", (unsigned long long)total);
    else {
        printf("lost %llu %d", (unsigned long long)total, most);
        for (int c = 0; c < 52; c++)
            if (inplay[c] && !(d.moved >> c & 1)) printf(" %d", c);
        printf("\n");
    }
    if (want_layers) {
        /* whole layers, in order of cards home and then of number, as Solver B lists them */
        for (int h = 0; h <= 52; h++)
            for (int id = 0; id < NLAYER; id++) {
                uint8_t hh[4];
                layer_home(id, hh);
                if (layer_seen[id] && home_total(hh) == h)
                    printf("layer %d %d %d %d %llu\n", hh[0], hh[1], hh[2], hh[3],
                           (unsigned long long)layer_count[id]);
            }
        printf("sublayers %llu %llu\n", (unsigned long long)sublayers, (unsigned long long)largest);
        printf("end\n");
    }
    wait_clear_all();
    table_clear(&t16); table_clear(&t32);
}

/* ---- finding a win: the layer with most cards home first ----

   The search visits layers again and again: a visit searches one layer from the positions
   waiting for it, and a layer that gets more positions after its visit is visited again.
   Every position reached is eventually searched, so the search also ends when the level is
   lost, but positions can be searched more than once, and only "decide" counts them.

   To give a line of moves, each waiting position keeps its parent: the visit it came from
   and the parent's place in that visit's table. A visit writes its starting positions, with
   their parents, to a file. A visit can be run again from that file and puts the same
   positions in the same places, so a path within it can be found by running it again with
   a parent for each position. Every code is kept at 32 bytes here. */

typedef struct { uint64_t code[4]; uint32_t pvisit, pindex; } Seed;
#define NOPARENT 0xFFFFFFFFu
static Seed *wseed[NLAYER];
static uint64_t wseedn[NLAYER], wseedcap[NLAYER];
static int *visit_layer; static uint64_t nvisit, visitcap;
static uint32_t *parent_of; static uint64_t parent_reserved;

static void seed_path(char *buf, uint64_t v) {
    snprintf(buf, 600, "%s/fc-c-%d-visit-%llu.bin", folder, (int)getpid(), (unsigned long long)v);
}
static void seed_push(int id, const Code *k, uint32_t pv, uint32_t pi) {
    if (wseedn[id] == wseedcap[id]) {
        wseedcap[id] = wseedcap[id] ? 2 * wseedcap[id] : 64;
        wseed[id] = realloc(wseed[id], wseedcap[id] * sizeof(Seed));
        if (!wseed[id]) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    Seed *s = &wseed[id][wseedn[id]++];
    memcpy(s->code, k->w, 32);
    s->pvisit = pv; s->pindex = pi;
    pending[id] = 1;
}

typedef struct { int cur, won, full; uint32_t visit, index; int card, dest; } Hunt;

static int hunt_visit(void *ctx, Pos *q, int card, int dest, const uint8_t *played, int n) {
    Hunt *h = ctx;
    (void)played; (void)n;
    if (home_total(q->home) == decksize) { h->won = 1; h->card = card; h->dest = dest; return 1; }
    Code k;
    encode(q, &k);
    int id = layer_id(q->home);
    if (id != h->cur) { seed_push(id, &k, h->visit, h->index); return 0; }
    if (table_add(&t32, k.w) < 0) h->full = 1;
    return 0;
}

/* Run visit v again, with a parent for each position, until position `target` is placed.
   Returns how many codes the path holds (written to `path`, starting position first) and
   sets *seed to the starting position's place in the visit's file. */
static uint64_t retrace_from;
static int retrace_cur;
static int retrace_visit(void *ctx, Pos *q, int card, int dest, const uint8_t *played, int n) {
    (void)ctx; (void)card; (void)dest; (void)played; (void)n;
    if (layer_id(q->home) != retrace_cur || home_total(q->home) == decksize) return 0;
    Code k;
    encode(q, &k);
    if (table_add(&t32, k.w) == 1) parent_of[t32.n - 1] = (uint32_t)retrace_from;
    return 0;
}
static uint64_t retrace(uint64_t v, uint32_t target, uint64_t **path, Seed *seed) {
    char fp[600];
    seed_path(fp, v);
    FILE *f = fopen(fp, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", fp); exit(2); }
    table_clear(&t32);
    Seed s;
    uint64_t nseeds = 0;
    Seed *seeds = 0;
    while (fread(&s, sizeof s, 1, f) == 1) {
        seeds = realloc(seeds, (nseeds + 1) * sizeof s);
        seeds[nseeds++] = s;
        if (table_add(&t32, s.code) == 1) parent_of[t32.n - 1] = NOPARENT;
    }
    fclose(f);
    retrace_cur = visit_layer[v];
    uint8_t home[4];
    layer_home(retrace_cur, home);
    Pos p;
    for (uint64_t i = 0; t32.n <= target && i < t32.n; i++) {
        decode(t32.keys + 4 * i, home, &p);
        retrace_from = i;
        successors(&p, retrace_visit, 0);
    }
    if (target >= t32.n) { fprintf(stderr, "retrace failed\n"); exit(2); }
    uint64_t len = 0;
    for (uint64_t i = target; i != NOPARENT; i = parent_of[i]) len++;
    *path = malloc(len * 32);
    uint64_t at = len, i = target, first = 0;
    for (; i != NOPARENT; i = parent_of[i]) { at--; memcpy(*path + 4 * at, t32.keys + 4 * i, 32); first = i; }
    *seed = seeds[first]; /* seeds are placed first, in the order of the file */
    free(seeds);
    return len;
}

/* The move from p to the position with code `to` (or to the won position, if to is 0). */
typedef struct { const uint64_t *to; int to_layer, found; } Step;
static void print_move(int card, int dest, const uint8_t *played, int n) {
    if (dest == TO_HOME) printf(" %d>h", card);
    else if (dest == TO_CELL) printf(" %d>c", card);
    else if (dest == TO_EMPTY) printf(" %d>e", card);
    else printf(" %d>%d", card, dest);
    for (int i = 0; i < n; i++) printf(" %d>h", played[i]);
}
static int step_visit(void *ctx, Pos *q, int card, int dest, const uint8_t *played, int n) {
    Step *st = ctx;
    int hit;
    if (!st->to) hit = home_total(q->home) == decksize;
    else {
        Code k;
        encode(q, &k);
        hit = !memcmp(k.w, st->to, 32) && layer_id(q->home) == st->to_layer;
    }
    if (!hit) return 0;
    print_move(card, dest, played, n);
    st->found = 1;
    return 1;
}

static void hunt(Pos *start) {
    uint8_t played[52];
    int n0 = autoplay(start, played);
    Hunt h = {0, 0, 0, 0, 0, 0, 0};
    uint64_t total = 0;
    nvisit = 0;
    if (home_total(start->home) == decksize) {
        printf("won 1");
        for (int i = 0; i < n0; i++) printf(" %d>h", played[i]);
        printf("\n");
        return;
    }
    Code k;
    encode(start, &k);
    seed_push(layer_id(start->home), &k, NOPARENT, NOPARENT);
    time_t t0 = time(0);
    while (!h.won && !h.full && (!budget || total < budget)) {
        int cur = -1, best = -1;
        for (int id = 0; id < NLAYER; id++)
            if (pending[id]) {
                uint8_t hh[4]; layer_home(id, hh);
                if (home_total(hh) > best) { best = home_total(hh); cur = id; }
            }
        if (cur < 0) break;
        if (nvisit == visitcap) {
            visitcap = visitcap ? 2 * visitcap : 1024;
            visit_layer = realloc(visit_layer, visitcap * sizeof *visit_layer);
        }
        uint64_t v = nvisit++;
        visit_layer[v] = cur;
        table_clear(&t32);
        char fp[600];
        seed_path(fp, v);
        FILE *f = fopen(fp, "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", fp); exit(2); }
        for (uint64_t i = 0; i < wseedn[cur]; i++)
            if (table_add(&t32, wseed[cur][i].code) == 1)
                fwrite(&wseed[cur][i], sizeof(Seed), 1, f);
        fclose(f);
        free(wseed[cur]); wseed[cur] = 0; wseedn[cur] = wseedcap[cur] = 0;
        pending[cur] = 0;
        uint8_t home[4];
        layer_home(cur, home);
        h.cur = cur;
        h.visit = (uint32_t)v;
        Pos p;
        for (uint64_t i = 0; i < t32.n && !h.won && !h.full; i++) {
            decode(t32.keys + 4 * i, home, &p);
            h.index = (uint32_t)i;
            successors(&p, hunt_visit, &h);
            if (layer_cap && t32.n > layer_cap) h.full = 1;
        }
        total += t32.n;
        if (getenv("SOLVER_C_PROGRESS") && (v % 1000 == 0 || t32.n > 1000000))
            fprintf(stderr, "visit %llu layer %d %d %d %d: %llu (total %llu, %lds)\n",
                    (unsigned long long)v, home[0], home[1], home[2], home[3],
                    (unsigned long long)t32.n, (unsigned long long)total, (long)(time(0) - t0));
    }
    if (h.won) {
        /* the move that wins, after the path to the position it is made from */
        /* Walk back from the position that wins, one visit at a time. */
        uint64_t **paths = 0, *lens = 0, nseg = 0;
        uint32_t *segv = 0, v = h.visit, idx = h.index;
        for (;;) {
            paths = realloc(paths, (nseg + 1) * sizeof *paths);
            lens = realloc(lens, (nseg + 1) * sizeof *lens);
            segv = realloc(segv, (nseg + 1) * sizeof *segv);
            Seed s;
            segv[nseg] = v;
            lens[nseg] = retrace(v, idx, &paths[nseg], &s);
            nseg++;
            if (s.pvisit == NOPARENT) break;
            v = s.pvisit; idx = s.pindex;
        }
        printf("won %llu", (unsigned long long)total);
        for (int i = 0; i < n0; i++) printf(" %d>h", played[i]);
        Pos p;
        for (uint64_t sgi = nseg; sgi-- > 0;) {
            uint8_t home[4];
            layer_home(visit_layer[segv[sgi]], home);
            for (uint64_t j = 0; j < lens[sgi]; j++) {
                decode(paths[sgi] + 4 * j, home, &p);
                Step st = {0, 0, 0};
                if (j + 1 < lens[sgi]) { st.to = paths[sgi] + 4 * (j + 1); st.to_layer = visit_layer[segv[sgi]]; }
                else if (sgi > 0) { st.to = paths[sgi - 1]; st.to_layer = visit_layer[segv[sgi - 1]]; }
                successors(&p, step_visit, &st);
                if (!st.found) { fprintf(stderr, "no move found\n"); exit(2); }
            }
        }
        printf("\n");
        for (uint64_t i = 0; i < nseg; i++) free(paths[i]);
        free(paths); free(lens); free(segv);
    } else if (h.full || (budget && total >= budget)) printf("unknown %llu\n", (unsigned long long)total);
    else printf("exhausted %llu\n", (unsigned long long)total);
    for (uint64_t v = 0; v < nvisit; v++) { char fp[600]; seed_path(fp, v); unlink(fp); }
    for (int id = 0; id < NLAYER; id++) {
        free(wseed[id]); wseed[id] = 0; wseedn[id] = wseedcap[id] = 0; pending[id] = 0;
    }
    table_clear(&t32);
}

/* The Microsoft deal, from the C runtime's rand() seeded with the deal number. */
static void microsoft(long deal) {
    uint32_t state = (uint32_t)deal;
    int deck[52], left = 52;
    for (int i = 0; i < 52; i++) deck[i] = i;
    for (int i = 0; i < 8; i++) dealtlen[i] = 0;
    for (int k = 0; k < 52; k++) {
        state = (state * 214013u + 2531011u) & 0x7fffffffu;
        int j = (int)((state >> 16) % (uint32_t)left);
        dealt[k % 8][dealtlen[k % 8]++] = (uint8_t)deck[j];
        deck[j] = deck[--left];
    }
}

int main(int argc, char **argv) {
    if (argc > 1) snprintf(folder, sizeof folder, "%s", argv[1]);
    layer_cap = argc > 2 ? strtoull(argv[2], 0, 10) : 0;
    if (argc > 3) memory_limit = strtoull(argv[3], 0, 10);
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "layers")) want_layers = 1;
        else if (!strcmp(argv[i], "horne")) horne = 1;
        else if (!strcmp(argv[i], "fine")) fine = 1;
        else if (!strncmp(argv[i], "win=", 4)) { win_mode = 1; budget = strtoull(argv[i] + 4, 0, 10); }
    }
    /* address space for each table: as many codes as physical memory could hold */
    uint64_t phys = (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t reserve = phys / 16 < (1ULL << 32) - 2 ? phys / 16 : (1ULL << 32) - 2;
    if (layer_cap && layer_cap + (1 << 20) < reserve) reserve = layer_cap + (1 << 20);
    table_init(&t16, 2, reserve);
    table_init(&t32, 4, reserve);
    if (win_mode) {
        parent_reserved = reserve;
        parent_of = mmap(0, reserve * 4, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (parent_of == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(2); }
    }
    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        memset(inplay, 0, sizeof inplay);
        memset(suit_in_play, 0, sizeof suit_in_play);
        if (!strcmp(tok, "ms")) {
            long deal = atol(strtok(0, " \t\n"));
            ncells = atoi(strtok(0, " \t\n"));
            nranks = 13; nsuits = 4; ncols = 8;
            microsoft(deal);
            for (int c = 0; c < 52; c++) inplay[c] = 1;
        } else if (!strcmp(tok, "deck")) {
            nranks = atoi(strtok(0, " \t\n"));
            nsuits = atoi(strtok(0, " \t\n"));
            ncols = atoi(strtok(0, " \t\n"));
            ncells = atoi(strtok(0, " \t\n"));
            int i = 0, bad = 0;
            dealtlen[0] = 0;
            while ((tok = strtok(0, " \t\n"))) {
                if (!strcmp(tok, "|")) { dealtlen[++i] = 0; continue; }
                if (dealtlen[i] == 7) bad = 1; else dealt[i][dealtlen[i]++] = (uint8_t)atoi(tok);
            }
            if (i + 1 != ncols || bad || ncols > MAXCOL || ncells > 7) {
                printf("error layout\n"); fflush(stdout); continue;
            }
        } else { printf("error input\n"); fflush(stdout); continue; }
        decksize = 0;
        for (int c = 0; c < 52; c++) bottom_of[c] = -1;
        for (int c = 0; c < 52; c++) suit_in_play[c % 4] |= inplay[c];
        Pos start;
        memset(&start, 0, sizeof start);
        for (int i = 0; i < ncols; i++) {
            if (dealtlen[i]) bottom_of[dealt[i][0]] = i;
            for (int j = 0; j < dealtlen[i]; j++) {
                inplay[dealt[i][j]] = 1; decksize++;
                start.col[i][j] = (uint8_t)dealt[i][j];
            }
            start.len[i] = (uint8_t)dealtlen[i];
        }
        if (win_mode) hunt(&start); else decide(&start);
        fflush(stdout);
    }
    return 0;
}
