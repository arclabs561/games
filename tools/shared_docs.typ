// What a paper shares with its game's README: the README's shared sections, which
// tools/sync_docs.py fills from <game>/docs/*.md and <game>/results.json, rendered with
// cmarker; and the number formats the paper uses for its own reads of results.json.
//
//   #import "/tools/shared_docs.typ": shared, fmt
//   #let readme = read("README.md")
//   #shared(readme, "open")
//
// Paths that start with "/" are relative to the repository root, which typst_vendor.py
// passes to the compiler (plain typst: --root .).
#import "@preview/cmarker:0.1.10"

// The Markdown between <!-- shared:name --> and <!-- /shared --> in a README.
#let section(readme, name) = {
  let start = "<!-- shared:" + name + " -->"
  let i = readme.position(start)
  assert(i != none, message: "the README has no shared section " + name)
  let rest = readme.slice(i + start.len())
  let j = rest.position("<!-- /shared -->")
  assert(j != none, message: "shared section " + name + " is not closed")
  rest.slice(0, j)
}

// A shared section as Typst content. Markdown tables come out in the house style, three
// horizontal rules and none vertical; Markdown says nothing about widths, so the paper gives
// the columns (a 1fr column for the widest text) and, if it wants, the alignment.
// Fragments may carry Typst-only text in <!--raw-typst ...--> comments, such as
// cross-references, which GitHub does not show.
//
// The table cmarker makes is rebuilt in the house style before it is placed, not by a show
// rule on table: a show rule that returns a new table nests one tagged Table inside another,
// which PDF/UA-1 rejects ("invalid table structure").
#let house-table(it, columns, align) = {
  let cells = it.children
  let head = ()
  if cells.len() > 0 and cells.first().func() == table.header {
    head = (cells.remove(0), table.hline(stroke: 0.4pt))
  }
  // labeled, so that a paper's own show rule for tables can tell this one apart
  [#table(
    columns: if columns == auto { it.columns } else { columns },
    align: align,
    table.hline(stroke: 0.6pt), ..head, ..cells, table.hline(stroke: 0.6pt),
  ) <shared-table>]
}
#let restyle-tables(c, columns, align) = {
  if type(c) != content { c }
  else if c.func() == table { house-table(c, columns, align) }
  else if c.has("children") { c.children.map(x => restyle-tables(x, columns, align)).join() }
  else { c }
}
// `math` renders $...$ in a fragment (cmarker's math callback); with the default, none, dollar
// signs are plain text, as they were before the option existed.
#let shared(readme, name, columns: auto, align: left, math: none) = restyle-tables(
  cmarker.render(section(readme, name), heading-labels: none, set-document-title: false, math: math),
  columns,
  align,
)

// A whole number with thousands separators, as sync_docs.py writes it: 12362536 -> 12,362,536.
#let fmt(n) = {
  let s = str(n)
  let k = s.len()
  s.clusters().enumerate().map(((i, c)) => if i > 0 and calc.rem(k - i, 3) == 0 { "," + c } else { c }).join()
}

// 14716328 -> "14.7 million", as sync_docs.py writes it.
#let million(n) = {
  let s = str(calc.round(n / 1e6, digits: 1))
  (if s.contains(".") { s } else { s + ".0" }) + " million"
}
