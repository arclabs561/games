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
   is an ace). A position is the multiset of columns, the set of free cells and the
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
   written to files (default 1 GB), and optionally "layers".

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

typedef struct { uint64_t *buf; uint64_t n, cap; int on_disk; } Wait;
static Wait wait16[NLAYER], wait32[NLAYER];
static uint64_t waiting_bytes, memory_limit = 1ULL << 30;
static char folder[512] = "/tmp";
static uint8_t pending[NLAYER];

static void wait_path(char *buf, int id, int words) {
    snprintf(buf, 600, "%s/fc-c-%d-%d-%d.bin", folder, (int)getpid(), id, words);
}
static void flush_all(void) {
    char path[600];
    for (int w = 0; w < 2; w++)
        for (int id = 0; id < NLAYER; id++) {
            Wait *q = w ? &wait32[id] : &wait16[id];
            int words = w ? 4 : 2;
            if (!q->n) continue;
            wait_path(path, id, words);
            FILE *f = fopen(path, "ab");
            if (!f || fwrite(q->buf, 8 * words, q->n, f) != q->n) {
                fprintf(stderr, "cannot write %s\n", path); exit(2);
            }
            fclose(f);
            q->on_disk = 1;
            waiting_bytes -= q->n * 8 * words;
            q->n = 0;
            free(q->buf); q->buf = 0; q->cap = 0;
        }
}
static void wait_add(int id, const Code *k) {
    int words = k->n <= 128 ? 2 : 4;
    Wait *q = words == 2 ? &wait16[id] : &wait32[id];
    if (q->n == q->cap) {
        q->cap = q->cap ? 2 * q->cap : 256;
        q->buf = realloc(q->buf, q->cap * 8 * words);
        if (!q->buf) { fprintf(stderr, "out of memory\n"); exit(2); }
    }
    memcpy(q->buf + q->n * words, k->w, 8 * words);
    q->n++;
    pending[id] = 1;
    waiting_bytes += 8 * words;
    if (waiting_bytes > memory_limit) flush_all();
}
static void wait_clear_all(void) {
    char path[600];
    for (int id = 0; id < NLAYER; id++) {
        for (int w = 0; w < 2; w++) {
            Wait *q = w ? &wait32[id] : &wait16[id];
            if (q->on_disk) { wait_path(path, id, w ? 4 : 2); unlink(path); }
            free(q->buf);
            memset(q, 0, sizeof *q);
        }
        pending[id] = 0;
    }
    waiting_bytes = 0;
}

/* ---- moves ---- */
static int safe(const Pos *p, int c) {
    if (p->home[suit_of(c)] != rank_of(c) - 1) return 0;
    int r = rank_of(c);
    if (r <= 2) return 1;
    for (int s = 0; s < 4; s++) {
        int d = 4 * (r - 2) + s;
        if (!inplay[d] || is_red(d) == is_red(c)) continue;
        if (p->home[s] < r - 1) return 0;
    }
    return 1;
}
static void autoplay(Pos *p, uint64_t *movedmask) {
    for (int again = 1; again;) {
        again = 0;
        for (int i = 0; i < ncols; i++)
            while (p->len[i] && safe(p, p->col[i][p->len[i] - 1])) {
                int c = p->col[i][--p->len[i]];
                p->home[suit_of(c)]++;
                *movedmask |= 1ULL << c;
                again = 1;
            }
        for (int a = 0; a < p->ncell; a++)
            if (safe(p, p->cell[a])) {
                int c = p->cell[a];
                p->home[suit_of(c)]++;
                *movedmask |= 1ULL << c;
                p->cell[a] = p->cell[--p->ncell];
                a--;
                again = 1;
            }
    }
}

/* ---- the search ---- */
static Table t16, t32;
static uint64_t layer_cap;
static int want_layers;

typedef struct { int id; uint64_t count; } LayerCount;
static LayerCount *layerlog; static int nlayerlog, layercap;

static int home_total(const uint8_t *h) { return h[0] + h[1] + h[2] + h[3]; }

/* Successor q of the position being expanded, in layer `cur`: 1 if q is won. */
static int offer(Pos *q, int cur, uint64_t *movedmask, int *full) {
    autoplay(q, movedmask);
    if (home_total(q->home) == decksize) return 1;
    Code k;
    encode(q, &k);
    int id = layer_id(q->home);
    if (id != cur) { wait_add(id, &k); return 0; }
    int r = k.n <= 128 ? table_add(&t16, k.w) : table_add(&t32, k.w);
    if (r < 0) *full = 1;
    return 0;
}

static void load_waiting(int id, int *full) {
    char path[600];
    for (int w = 0; w < 2; w++) {
        int words = w ? 4 : 2;
        Wait *q = w ? &wait32[id] : &wait16[id];
        Table *t = w ? &t32 : &t16;
        if (q->on_disk) {
            wait_path(path, id, words);
            FILE *f = fopen(path, "rb");
            if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
            uint64_t buf[4 * 4096];
            size_t got;
            while ((got = fread(buf, 8 * words, 4096, f)) > 0)
                for (size_t i = 0; i < got; i++)
                    if (table_add(t, buf + i * words) < 0) *full = 1;
            fclose(f);
            unlink(path);
            q->on_disk = 0;
        }
        for (uint64_t i = 0; i < q->n; i++)
            if (table_add(t, q->buf + i * words) < 0) *full = 1;
        waiting_bytes -= q->n * 8 * words;
        free(q->buf);
        memset(q, 0, sizeof *q);
    }
    pending[id] = 0;
}

/* Expand position p (of layer cur); 1 if a successor is won. */
static int expand(const Pos *p, int cur, uint64_t *movedmask, int *full) {
    Pos q;
    int empty = -1;
    for (int i = 0; i < ncols; i++) if (!p->len[i]) { empty = i; break; }
    /* sources: column tops, then cells */
    for (int src = 0; src < ncols + p->ncell; src++) {
        int c, fromcol = src < ncols;
        if (fromcol) { if (!p->len[src]) continue; c = p->col[src][p->len[src] - 1]; }
        else c = p->cell[src - ncols];
        uint64_t bit = 1ULL << c;
#define LIFT() do { q = *p; if (fromcol) q.len[src]--; \
                    else { q.cell[src - ncols] = q.cell[--q.ncell]; } } while (0)
        if (p->home[suit_of(c)] == rank_of(c) - 1) {
            LIFT(); q.home[suit_of(c)]++; *movedmask |= bit;
            if (offer(&q, cur, movedmask, full)) return 1;
        }
        for (int t = 0; t < ncols; t++) {
            if (t == src && fromcol) continue;
            if (!p->len[t] || !fits(c, p->col[t][p->len[t] - 1])) continue;
            LIFT(); q.col[t][q.len[t]++] = (uint8_t)c; *movedmask |= bit;
            if (offer(&q, cur, movedmask, full)) return 1;
        }
        if (empty >= 0 && !(fromcol && p->len[src] == 1)) {
            LIFT(); q.col[empty][0] = (uint8_t)c; q.len[empty] = 1; *movedmask |= bit;
            if (offer(&q, cur, movedmask, full)) return 1;
        }
        if (fromcol && p->ncell < ncells) {
            LIFT(); q.cell[q.ncell++] = (uint8_t)c; *movedmask |= bit;
            if (offer(&q, cur, movedmask, full)) return 1;
        }
#undef LIFT
    }
    return 0;
}

static void solve(Pos *start) {
    uint64_t movedmask = 0, total = 0;
    int most = 0, full = 0, won = 0;
    nlayerlog = 0;
    autoplay(start, &movedmask);
    if (home_total(start->home) == decksize) won = 1;
    else {
        Code k;
        encode(start, &k);
        wait_add(layer_id(start->home), &k);
    }
    time_t t0 = time(0);
    while (!won && !full) {
        /* the waiting layer with fewest cards home */
        int cur = -1, best = 99;
        for (int id = 0; id < NLAYER; id++)
            if (pending[id]) {
                uint8_t h[4]; layer_home(id, h);
                if (home_total(h) < best) { best = home_total(h); cur = id; }
            }
        if (cur < 0) break;
        uint8_t home[4];
        layer_home(cur, home);
        if (home_total(home) > most) most = home_total(home);
        table_clear(&t16); table_clear(&t32);
        load_waiting(cur, &full);
        Pos p;
        /* Each table is read in order of insertion; both grow while they are read. */
        uint64_t i16 = 0, i32 = 0;
        while (!won && !full && (i16 < t16.n || i32 < t32.n)) {
            if (i16 < t16.n) decode(t16.keys + 2 * i16++, home, &p);
            else decode(t32.keys + 4 * i32++, home, &p);
            if (expand(&p, cur, &movedmask, &full)) won = 1;
            if (layer_cap && t16.n + t32.n > layer_cap) full = 1;
        }
        uint64_t count = t16.n + t32.n;
        total += count;
        if (nlayerlog == layercap) {
            layercap = layercap ? 2 * layercap : 1024;
            layerlog = realloc(layerlog, layercap * sizeof *layerlog);
        }
        layerlog[nlayerlog++] = (LayerCount){cur, count};
        if (getenv("SOLVER_C_PROGRESS"))
            fprintf(stderr, "layer %d %d %d %d: %llu (total %llu, waiting %llu MB, %lds)\n",
                    home[0], home[1], home[2], home[3], (unsigned long long)count,
                    (unsigned long long)total, (unsigned long long)(waiting_bytes >> 20),
                    (long)(time(0) - t0));
    }
    if (won) printf("won %llu %d\n", (unsigned long long)total, decksize);
    else if (full) printf("unknown %llu\n", (unsigned long long)total);
    else {
        printf("lost %llu %d", (unsigned long long)total, most);
        for (int c = 0; c < 52; c++)
            if (inplay[c] && !(movedmask >> c & 1)) printf(" %d", c);
        printf("\n");
    }
    if (want_layers) {
        for (int i = 0; i < nlayerlog; i++) {
            uint8_t h[4]; layer_home(layerlog[i].id, h);
            printf("layer %d %d %d %d %llu\n", h[0], h[1], h[2], h[3],
                   (unsigned long long)layerlog[i].count);
        }
        printf("end\n");
    }
    wait_clear_all();
    table_clear(&t16); table_clear(&t32);
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
    for (int i = 4; i < argc; i++) if (!strcmp(argv[i], "layers")) want_layers = 1;
    /* address space for each table: as many codes as physical memory could hold */
    uint64_t phys = (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE);
    uint64_t reserve = phys / 16 < (1ULL << 32) - 2 ? phys / 16 : (1ULL << 32) - 2;
    if (layer_cap && layer_cap + (1 << 20) < reserve) reserve = layer_cap + (1 << 20);
    table_init(&t16, 2, reserve);
    table_init(&t32, 4, reserve);
    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        memset(inplay, 0, sizeof inplay);
        if (!strcmp(tok, "ms")) {
            long deal = atol(strtok(0, " \t\n"));
            ncells = atoi(strtok(0, " \t\n"));
            nranks = 13; nsuits = 4; ncols = 8;
            microsoft(deal);
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
        /* cards absent from a small deck count as home if below every card in play */
        for (int s = 0; s < 4; s++) start.home[s] = 0;
        solve(&start);
        fflush(stdout);
    }
    return 0;
}
