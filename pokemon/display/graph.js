// Force-directed drawing of display/graph.json (written by ../graph.py), using D3 v7.
const KIND = { 2: "super", 0.5: "resisted", 0: "immune" };
const RADIUS = 24;

d3.json("graph.json").then(({ nodes, links }) => {
  const svg = d3.select("#chart");
  const width = +svg.attr("width");
  const height = +svg.attr("height");
  const arrows = links.filter((l) => l.source !== l.target);

  svg.append("defs").selectAll("marker")
    .data(Object.values(KIND))
    .join("marker")
    .attr("id", (k) => `arrow-${k}`)
    .attr("viewBox", "0 -5 10 10")
    .attr("refX", 9)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto")
    .append("path")
    .attr("class", (k) => `head ${k}`)
    .attr("d", "M0,-5L10,0L0,5");

  const link = svg.append("g").selectAll("path")
    .data(arrows)
    .join("path")
    .attr("class", (l) => `link ${KIND[l.multiplier]}`)
    .attr("marker-end", (l) => `url(#arrow-${KIND[l.multiplier]})`);

  const node = svg.append("g").selectAll("g")
    .data(nodes)
    .join("g")
    .attr("class", "node");
  node.append("circle").attr("r", RADIUS);
  node.append("text").attr("dy", "0.35em").text((d) => d.id);
  node.append("title").text((d) => matchups(d.id, links));

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(arrows).id((d) => d.id).distance(170).strength(0.05))
    .force("charge", d3.forceManyBody().strength(-500))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collide", d3.forceCollide(30))
    .on("tick", () => {
      link.attr("d", arc);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

  node.call(d3.drag()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }));
});

// A gentle arc from the edge of one circle to the edge of the other, so that the two
// directions between a pair of types bend to opposite sides instead of overlapping.
function arc({ source: s, target: t }) {
  const dx = t.x - s.x;
  const dy = t.y - s.y;
  const dist = Math.hypot(dx, dy) || 1;
  const gap = (RADIUS + 3) / dist;
  const [x1, y1] = [s.x + dx * gap, s.y + dy * gap];
  const [x2, y2] = [t.x - dx * gap, t.y - dy * gap];
  return `M${x1},${y1}A${dist * 1.6},${dist * 1.6} 0 0,1 ${x2},${y2}`;
}

// A tooltip listing what a type does to others and what others do to it.
function matchups(type, links) {
  const id = (end) => (typeof end === "object" ? end.id : end);
  const list = (rows, end, m) =>
    rows.filter((l) => l.multiplier === m).map((l) => id(l[end])).join(", ") || "none";
  const out = links.filter((l) => id(l.source) === type);
  const into = links.filter((l) => id(l.target) === type);
  return [
    type,
    `2× against: ${list(out, "target", 2)}`,
    `½× against: ${list(out, "target", 0.5)}`,
    `no effect on: ${list(out, "target", 0)}`,
    `takes 2× from: ${list(into, "source", 2)}`,
  ].join("\n");
}
