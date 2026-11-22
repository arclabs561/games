// Check whole-hand target thresholds through all 40 nonzero cards.
// Run through report.py whole_deck, or compile with c++ -std=c++20 -O2.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
using Bits = uint64_t;
constexpr int B = 63;
struct Hand {
  int code;
  vector<int> cards;
  Bits known = 0;
  array<int, 13> mod{};
};
vector<Hand> hands;
vector<int> ids(9765625, -1);
array<int, 10> powers;
map<int, vector<int>> exact_small;
map<pair<int, int>, bool> memo;
struct Row {
  int made = 0;
  array<int, 13> least{};
  int exceptions = 0;
  vector<int> first;
};
Row rows[152][12];
int first_target = 1, last_target = 30;

void generate(vector<int> &cards, int n, int low, int code) {
  if (int(cards.size()) == n) {
    ids[code] = hands.size();
    hands.push_back({code, cards});
    return;
  }
  for (int v = low; v <= 10; v++) {
    if (cards.size() >= 4 && cards[cards.size() - 4] == v)
      continue;
    cards.push_back(v);
    generate(cards, n, v, code + powers[v - 1]);
    cards.pop_back();
  }
}
// Each split is a submultiset / complement. Fix its orientation by code.
template <class Fn> bool split(const Hand &h, Fn fn) {
  array<int, 12> subs{};
  array<int, 12> count{};
  int kinds = 0;
  for (int v : h.cards) {
    if (!kinds || subs[kinds - 1] != powers[v - 1])
      subs[kinds++] = powers[v - 1];
    ++count[kinds - 1];
  }
  auto walk = [&](auto &&self, int i, int code) -> bool {
    if (i == kinds)
      return code > 0 && code <= h.code - code && code < h.code &&
             fn(ids[code], ids[h.code - code]);
    for (int j = 0; j <= count[i]; j++)
      if (self(self, i + 1, code + j * subs[i]))
        return true;
    return false;
  };
  return walk(walk, 0, 0);
}
void add_value(Bits &out, int v) {
  if (v <= B)
    out |= Bits(1) << v;
}
// All nonnegative leaf expressions can be normalized to nonnegative magnitudes:
// combine by sum, absolute difference or product. Each constructed value is
// attainable.
Bits lower(const Hand &h) {
  if (h.cards.size() == 1)
    return Bits(1) << h.cards[0];
  Bits out = 0;
  split(h, [&](int i, int j) {
    Bits a = hands[i].known, b = hands[j].known;
    for (Bits x = a; x; x &= x - 1) {
      int u = countr_zero(x);
      for (Bits y = b; y; y &= y - 1) {
        int v = countr_zero(y);
        add_value(out, u + v);
        add_value(out, abs(u - v));
        add_value(out, u * v);
      }
    }
    // Any omitted values only weaken this constructive first pass.
    return (out & 0x7fffffffULL) == 0x7fffffffULL;
  });
  return out;
}
int residues(int id, int m) {
  auto &h = hands[id];
  int &save = h.mod[m];
  if (save)
    return save;
  if (h.cards.size() == 1)
    return save = 1 << (h.cards[0] % m);
  int out = 0, full = (1 << m) - 1;
  for (Bits x = h.known; x; x &= x - 1)
    out |= 1 << (countr_zero(x) % m);
  if (out == full)
    return save = out;
  split(h, [&](int i, int j) {
    int a = residues(i, m), b = residues(j, m);
    for (int x = 0; x < m; x++)
      if (a >> x & 1)
        for (int y = 0; y < m; y++)
          if (b >> y & 1)
            for (int z : {x + y, x - y + m, y - x + m, x * y})
              out |= 1 << (z % m);
    return out == full;
  });
  return save = out;
}
const vector<int> &values(int id) {
  if (auto it = exact_small.find(id); it != exact_small.end())
    return it->second;
  auto &h = hands[id];
  set<int> out;
  if (h.cards.size() == 1)
    out.insert(h.cards[0]);
  else
    split(h, [&](int i, int j) {
      for (int a : values(i))
        for (int b : values(j)) {
          out.insert(a + b);
          out.insert(abs(a - b));
          out.insert(a * b);
        }
      return false;
    });
  return exact_small.emplace(id, vector<int>(out.begin(), out.end()))
      .first->second;
}
bool makes(int id, int t) {
  auto &h = hands[id];
  if (t <= B && ((h.known >> t) & 1))
    return true;
  if (h.cards.size() <= 4) {
    const auto &v = values(id);
    return binary_search(v.begin(), v.end(), t);
  }
  auto key = make_pair(id, t);
  if (auto it = memo.find(key); it != memo.end())
    return it->second;
  bool ok = split(h, [&](int i, int j) {
    if (hands[i].cards.size() > hands[j].cards.size())
      swap(i, j);
    for (int x : values(i)) {
      if (x == 0 && t == 0)
        return true;
      if ((x && t % x == 0 && makes(j, t / x)) || makes(j, abs(t - x)) ||
          makes(j, t + x))
        return true;
    }
    return false;
  });
  return memo[key] = ok;
}
uint32_t targets_mask;
vector<uint32_t> can(9765625);

// An expression using at most two leaves nonzero mod m is 0, a signed
// leaf, a signed sum of both, or their signed product mod m.
uint32_t easy_blocks(const array<int, 10> &counts) {
  uint32_t out = 0;
  for (int m = 2; m <= 12; m++) {
    int loud[2] = {0, 0}, k = 0;
    for (int v = 1; v <= 10 && k <= 2; v++)
      if (v % m)
        for (int j = 0; j < counts[v - 1] && k <= 2; j++) {
          if (k < 2)
            loud[k] = v;
          ++k;
        }
    if (k > 2)
      continue;
    int possible = 1, a = loud[0], b = loud[1];
    for (int x : {a, b, a + b, a - b, a * b})
      possible |= 1 << (x % m + m) % m | 1 << (-x % m + m) % m;
    for (int t = first_target; t <= last_target; t++)
      if (!(possible >> (t % m) & 1))
        out |= uint32_t(1) << (t - first_target);
  }
  return out;
}

void extend() {
  long long considered = 0, yes = 0, no = 0, unknown = 0;
  for (int code = 1; code < 9765625; code++) {
    array<int, 10> counts{};
    int n = 0;
    for (int v = 0, k = code; v < 10; v++, k /= 5)
      n += (counts[v] = k % 5);
    if (n < 12)
      continue;
    considered++;
    uint32_t made = 0;
    if (counts[0])
      made |= can[code - 1]; // one spare 1
    for (int v = 0; v < 10 && made != targets_mask; v++) {
      if (counts[v] >= 2)
        made |= can[code - 2 * powers[v]];
      if (v && counts[v] && counts[v - 1])
        made |= can[code - powers[v] - powers[v - 1]];
    }
    can[code] = made;
    yes += popcount(made);
    if (made == targets_mask)
      continue;
    uint32_t blocked = easy_blocks(counts);
    if (made & blocked)
      throw logic_error("witness overlaps a blocker");
    no += popcount(blocked);
    uint32_t rem = targets_mask & ~(blocked | made);
    if (rem && unknown < 15) {
      cerr << "unresolved:";
      for (int v = 0; v < 10; v++)
        for (int r = 0; r < counts[v]; r++)
          cerr << " " << v + 1;
      cerr << " target bits " << rem << "\n";
    }
    unknown += popcount(rem);
  }
  cout << "coverage " << considered << " " << yes << " " << no << " " << unknown
       << "\n";
  if (unknown || yes + no != (last_target - first_target + 1) * considered)
    throw logic_error("incomplete accounting");
}

int main(int argc, char **argv) {
  if (argc != 1 && (argc != 4 || string(argv[1]) != "--range"))
    return 2;
  if (argc == 4) {
    first_target = stoi(argv[2]);
    last_target = stoi(argv[3]);
    if (first_target < 1 || last_target > 151 || first_target > last_target ||
        last_target - first_target >= 30)
      return 2;
  }
  targets_mask = (uint32_t(1) << (last_target - first_target + 1)) - 1;
  powers[0] = 1;
  for (int i = 1; i < 10; i++)
    powers[i] = powers[i - 1] * 5;
  vector<int> cards;
  for (int n = 1; n <= 11; n++) {
    size_t start = hands.size();
    generate(cards, n, 1, 0);
    for (size_t i = start; i < hands.size(); i++) {
      auto &h = hands[i];
      h.known = lower(h);
      for (int t = first_target; t <= last_target; t++) {
        auto &r = rows[t][n];
        if (t <= B && (h.known >> t & 1)) {
          ++r.made;
          if (n >= 10)
            can[h.code] |= uint32_t(1) << (t - first_target);
          continue;
        }
        int m = 2;
        for (; m <= 12; m++)
          if (!(residues(i, m) >> (t % m) & 1))
            break;
        if (m <= 12) {
          ++r.least[m];
          continue;
        }
        if (makes(i, t)) {
          ++r.made;
          if (n >= 10)
            can[h.code] |= uint32_t(1) << (t - first_target);
          continue;
        }
        if (!r.exceptions)
          r.first = h.cards;
        ++r.exceptions;
      }
    }
    cerr << "size " << n << " finished (" << hands.size() - start
         << " multisets)\n";
  }
  for (int t = first_target; t <= last_target; t++) {
    int s = 1, total = 0;
    set<int> mods;
    for (int n = 1; n <= 11; n++)
      if (rows[t][n].exceptions)
        s = n + 1;
    for (int n = s; n <= 11; n++) {
      total += rows[t][n].made;
      for (int m = 2; m <= 12; m++)
        if (rows[t][n].least[m])
          mods.insert(m);
    }
    cout << t << " " << s << " " << total << " |";
    for (int m : mods)
      cout << " " << m;
    cout << " |";
    for (int n = 1; n <= 11; n++)
      cout << " " << rows[t][n].exceptions;
    cout << " |";
    for (int x : rows[t][s - 1].first)
      cout << " " << x;
    cout << "\n";
  }
  extend();
}
