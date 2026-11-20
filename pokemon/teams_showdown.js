// Runs many Pokémon Showdown battles in one process, for teams_showdown.py.
//
// Usage: node teams_showdown.js <Showdown's dist/sim directory>
//
// Reads one JSON object per line on stdin:
//   {"id": 7, "start": {"formatid": "gen9customgame", "seed": "sodium,..."},
//    "p1": [sets], "p2": [sets]}       starts battle 7
//   {"id": 7, "write": ">p1 move 2"}   sends a line to battle 7's stream
// and writes one JSON object per line on stdout: {"id": 7, "chunk": "..."} for each chunk
// the battle stream emits (updates, side updates with requests, and the end).
"use strict";

const path = require("path");
const readline = require("readline");

const { BattleStream } = require(path.join(process.argv[2], "battle-stream"));
const streams = new Map();

function emit(id, chunk) {
  process.stdout.write(JSON.stringify({ id, chunk }) + "\n");
}

async function pump(id, stream) {
  for await (const chunk of stream) {
    emit(id, chunk);
    if (chunk.startsWith("end")) break;
  }
  streams.delete(id);
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.start) {
    const stream = new BattleStream({ debug: false });
    streams.set(msg.id, stream);
    pump(msg.id, stream);
    stream.write(`>start ${JSON.stringify(msg.start)}`);
    stream.write(`>player p1 ${JSON.stringify({ name: "p1", team: msg.p1 })}`);
    stream.write(`>player p2 ${JSON.stringify({ name: "p2", team: msg.p2 })}`);
  } else if (streams.has(msg.id)) {
    streams.get(msg.id).write(msg.write);
  }
});
rl.on("close", () => {
  for (const s of streams.values()) s.writeEnd();
});
