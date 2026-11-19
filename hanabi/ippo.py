"""Plan H4: independent PPO with a recurrent policy on the two-color game, in JaxMARL.

    uv run --group hanabi python -m hanabi.ippo train --ent-coef 0.05 --seeds 1-8 \
        --steps 2e8 --out OUT [--deadline-minutes 200]
    uv run --group hanabi python -m hanabi.ippo evaluate --runs RUN_A RUN_B \
        --deals 400000-401999 --out OUT
    uv run --group hanabi python -m hanabi.ippo smoke      # a few seconds on a CPU

`train` trains one policy per seed, all seeds side by side in one process (`jax.vmap` over
seeds), and saves each seed's parameters and training curve. `evaluate` loads the policies
of one or more training runs (folders, or `hf:runs/h4/<run>` in the private dataset), plays
every ordered pair of policies on the test deals, greedy and sampled, and writes every
action and score; then `hanabi.h4.summarize` replays every game in `rules.py` and writes
the summary.

The algorithm is the recurrent IPPO of JaxMARL's baseline `baselines/IPPO/ippo_rnn_hanabi.py`
(v0.2.0, Apache 2.0), which builds on PureJaxRL, written again here because the baseline is
not part of the `jaxmarl` package. All agents share one network, as in the baseline and in
Forkel et al. (2025). Settings are the baseline's, except for these, which plan H4 and its
amendment fix:

- an LSTM in place of the baseline's GRU, and λ = 0.9 for generalized advantage estimation;
- the actor loss and the entropy bonus are averaged over the turns where the agent acts.
  The agent whose turn it is not has one legal action, contributes nothing to either, and
  in the baseline halves both with two players. Forkel et al. mask it out in the same way,
  so α means the same as in their paper;
- the advantage uses the done flag of the step itself. (The baseline file at v0.2.0 does
  not run as written: its transition record is built with one field missing.)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from hanabi.jax_bridge import H4_RULES, make_env, max_turns, to_jaxmarl
from hanabi.rules import seeded_deck

T0 = time.time()  # when the module was imported: the setup time inside the job starts here


@dataclass(frozen=True)
class Config:
    """PPO settings. Defaults: JaxMARL's `ippo_rnn_hanabi.yaml`, with λ = 0.9 (plan H4)."""

    ent_coef: float = 0.01
    lr: float = 5e-4
    num_envs: int = 1024
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    fc_dim: int = 128
    hidden_dim: int = 128
    total_steps: float = 2e8  # environment steps (turns) per seed

    @property
    def batch_steps(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_updates(self) -> int:
        return int(self.total_steps // self.batch_steps)


def parse_seeds(text: str) -> list[int]:
    """'1-8', '0,3,5' or '0,9-15'."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


# ---------------------------------------------------------------------------- network


def build_network(num_actions: int, cfg: Config):
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    from flax.linen.initializers import constant, orthogonal

    class ScannedLSTM(nn.Module):
        @partial(
            nn.scan,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},  # ty: ignore[invalid-argument-type]
        )
        @nn.compact
        def __call__(self, carry, x):
            ins, resets = x
            zeros = (jnp.zeros_like(carry[0]), jnp.zeros_like(carry[1]))
            carry = jax.tree.map(lambda z, c: jnp.where(resets[:, None], z, c), zeros, carry)
            carry, y = nn.OptimizedLSTMCell(features=ins.shape[1])(carry, ins)
            return carry, y

    class ActorCritic(nn.Module):
        @nn.compact
        def __call__(self, carry, x):
            obs, dones, avail = x
            emb = nn.Dense(cfg.fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
                obs
            )
            emb = nn.relu(emb)
            carry, emb = ScannedLSTM()(carry, (emb, dones))
            a = nn.Dense(cfg.hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(emb)
            a = nn.relu(a)
            a = nn.Dense(num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
            logits = a - (1 - avail) * 1e10
            c = nn.Dense(cfg.fc_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(emb)
            c = nn.relu(c)
            c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
            return carry, logits, jnp.squeeze(c, axis=-1)

    return ActorCritic()


def zero_carry(batch: int, hidden: int):
    import jax.numpy as jnp

    return (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden)))


def log_probs(logits):
    import jax

    return jax.nn.log_softmax(logits, axis=-1)


def entropy(logits, avail):
    import jax.numpy as jnp

    lp = log_probs(logits)
    return -jnp.sum(jnp.where(avail > 0, jnp.exp(lp) * lp, 0.0), axis=-1)


# ---------------------------------------------------------------------------- training


def make_trainer(cfg: Config, rules=H4_RULES):
    """Functions `init(key) -> runner` and `run(runner, n) -> (runner, metrics)` for one
    seed; `run` makes n PPO updates. Both are meant to be vmapped over seeds."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    env = make_env(rules)
    agents = env.agents
    n_agents, n_envs = len(agents), cfg.num_envs
    n_actors = n_agents * n_envs
    n_actions = env.action_space(agents[0]).n
    obs_dim = env.observation_space(agents[0]).shape
    network = build_network(n_actions, cfg)
    n_updates = cfg.num_updates

    def schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / n_updates
        return cfg.lr * frac

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )

    def batchify(x: dict):
        return jnp.stack([x[a] for a in agents]).reshape((n_actors, -1))

    def init(key):
        key, k_net, k_env = jax.random.split(key, 3)
        x = (
            jnp.zeros((1, n_envs, obs_dim)),
            jnp.zeros((1, n_envs), dtype=bool),
            jnp.ones((1, n_envs, n_actions)),
        )
        params = network.init(k_net, zero_carry(n_envs, cfg.hidden_dim), x)
        state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(k_env, n_envs))
        done = jnp.zeros(n_actors, dtype=bool)
        ep = jnp.zeros(n_actors)  # the score so far of each actor's current game
        runner = (state, env_state, obs, done, zero_carry(n_actors, cfg.hidden_dim), ep, key)
        # fix every dtype, so that the first chunk compiles to the same program as the rest
        return jax.tree.map(lambda x: jnp.asarray(x).astype(jnp.asarray(x).dtype), runner)

    def env_step(runner, _):
        state, env_state, last_obs, last_done, carry, ep, key = runner
        key, k_act, k_env = jax.random.split(key, 3)
        avail = batchify(jax.vmap(env.get_legal_moves)(env_state))
        obs = batchify(last_obs)
        carry, logits, value = network.apply(
            state.params, carry, (obs[None], last_done[None], avail[None])
        )
        logits, value = logits[0], value[0]
        action = jax.random.categorical(k_act, logits)
        logp = jnp.take_along_axis(log_probs(logits), action[:, None], axis=-1)[:, 0]
        env_act = {a: action.reshape(n_agents, n_envs)[i] for i, a in enumerate(agents)}
        obs2, env_state, reward, done, _ = jax.vmap(env.step)(
            jax.random.split(k_env, n_envs), env_state, env_act
        )
        done = batchify(done)[:, 0].astype(bool)
        reward = batchify(reward)[:, 0]
        ep = ep + reward
        transition = {
            "reset": last_done,  # this observation starts a new game
            "done": done,  # the game ended with this step
            "action": action,
            "value": value,
            "reward": reward,
            "logp": logp,
            "obs": obs,
            "avail": avail,
            "acting": 1.0 - avail[:, -1],  # "no move" is legal only when it is not your turn
            "ep_return": ep,
            "ep_end": done.astype(jnp.float32),
        }
        ep = jnp.where(done, 0.0, ep)
        return (state, env_state, obs2, done, carry, ep, key), transition

    def loss_fn(params, carry0, batch, adv, targets):
        _, logits, value = network.apply(
            params, carry0, (batch["obs"], batch["reset"], batch["avail"])
        )
        logp = jnp.take_along_axis(log_probs(logits), batch["action"][..., None], axis=-1)[..., 0]
        clipped = batch["value"] + (value - batch["value"]).clip(-cfg.clip_eps, cfg.clip_eps)
        v_loss = 0.5 * jnp.maximum((value - targets) ** 2, (clipped - targets) ** 2).mean()
        ratio = jnp.exp(logp - batch["logp"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surrogate = jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
        )
        acting = batch["acting"]
        n_acting = jnp.maximum(acting.sum(), 1.0)
        actor_loss = -(surrogate * acting).sum() / n_acting
        ent = (entropy(logits, batch["avail"]) * acting).sum() / n_acting
        total = actor_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
        kl = (((ratio - 1) - jnp.log(ratio)) * acting).sum() / n_acting
        return total, (v_loss, actor_loss, ent, kl)

    def update(runner, _):
        carry0 = runner[4]
        runner, traj = jax.lax.scan(env_step, runner, None, cfg.num_steps)
        state, env_state, last_obs, last_done, carry, ep, key = runner
        avail = jnp.ones((n_actors, n_actions))
        _, _, last_val = network.apply(
            state.params, carry, (batchify(last_obs)[None], last_done[None], avail[None])
        )
        last_val = last_val[0]

        def gae_step(acc, t):
            gae, next_value = acc
            alive = 1.0 - t["done"]
            delta = t["reward"] + cfg.gamma * next_value * alive - t["value"]
            gae = delta + cfg.gamma * cfg.gae_lambda * alive * gae
            return (gae, t["value"]), gae

        _, adv = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(last_val), last_val),
            {k: traj[k] for k in ("done", "reward", "value")},
            reverse=True,
            unroll=16,
        )
        targets = adv + traj["value"]
        keep = ("reset", "action", "value", "logp", "obs", "avail", "acting")
        batch = {k: traj[k] for k in keep}

        def epoch(carry_e, _):
            state, key = carry_e
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, n_actors)

            def split(x):  # time x actors x ... -> minibatches x time x actors/mb x ...
                x = jnp.take(x, perm, axis=1)
                x = x.reshape((x.shape[0], cfg.num_minibatches, -1) + x.shape[2:])
                return jnp.swapaxes(x, 0, 1)

            def split0(x):  # actors x ... -> minibatches x actors/mb x ...
                x = jnp.take(x, perm, axis=0)
                return x.reshape((cfg.num_minibatches, -1) + x.shape[1:])

            mbs = (
                jax.tree.map(split0, carry0),
                jax.tree.map(split, batch),
                split(adv),
                split(targets),
            )

            def minibatch(state, mb):
                c0, b, a, tg = mb
                grads, aux = jax.grad(loss_fn, has_aux=True)(state.params, c0, b, a, tg)
                return state.apply_gradients(grads=grads), aux

            state, aux = jax.lax.scan(minibatch, state, mbs)
            return (state, key), aux

        (state, key), aux = jax.lax.scan(epoch, (state, key), None, cfg.update_epochs)
        ends = traj["ep_end"].sum()
        metrics = {
            "return": (traj["ep_return"] * traj["ep_end"]).sum() / jnp.maximum(ends, 1.0),
            "games": ends / n_agents,
            "value_loss": aux[0].mean(),
            "actor_loss": aux[1].mean(),
            "entropy": aux[2].mean(),
            "approx_kl": aux[3].mean(),
        }
        return (state, env_state, last_obs, last_done, carry, ep, key), metrics

    def run(runner, n: int):
        return jax.lax.scan(update, runner, None, n)

    return init, run, network


def chunk_size(n_updates: int, target: int) -> int:
    """The largest divisor of n_updates that is at most target, so every chunk compiles
    to the same program."""
    return max(d for d in range(1, min(target, n_updates) + 1) if n_updates % d == 0)


def save_params(out: Path, seeds: list[int], params) -> None:
    import jax
    from flax import serialization

    for i, s in enumerate(seeds):
        one = jax.tree.map(lambda x, i=i: np.asarray(x[i]), params)
        (out / f"params-seed{s}.msgpack").write_bytes(serialization.to_bytes(one))


def train(cfg: Config, seeds: list[int], out: Path, chunk: int = 20, deadline: float = 0) -> dict:
    """Train one policy per seed, side by side. Seed s starts from `PRNGKey(s)`. Saves
    the parameters and curves after every chunk of updates; stops early, and says so, if
    another chunk would pass `deadline` seconds after the module was imported."""
    import jax
    import jax.numpy as jnp

    out.mkdir(parents=True, exist_ok=True)
    init, run, _ = make_trainer(cfg)
    n = chunk_size(cfg.num_updates, chunk)
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    t_start = time.time()
    runner = jax.jit(jax.vmap(init))(keys)
    run_chunk = jax.jit(jax.vmap(partial(run, n=n)))
    curves: dict[str, list] = {}
    record = {
        "config": asdict(cfg),
        "seeds": seeds,
        "updates_planned": cfg.num_updates,
        "chunk": n,
        "device": str(jax.devices()[0]),
        "import_seconds": round(t_start - T0, 1),
        "chunks": [],
    }
    done = 0
    while done < cfg.num_updates:
        t0 = time.time()
        runner, metrics = run_chunk(runner)
        metrics = jax.device_get(metrics)
        dt = time.time() - t0
        done += n
        for k, v in metrics.items():
            curves.setdefault(k, []).append(np.asarray(v))  # seeds x n
        record["chunks"].append(round(dt, 2))
        record["updates_done"] = done
        record["steps_per_seed"] = done * cfg.batch_steps
        steady = record["chunks"][1:] or record["chunks"]
        # the rounded chunk times are kept; a tiny run's chunk may round to 0 seconds
        record["steps_per_second_all_seeds"] = round(
            len(seeds) * n * cfg.batch_steps / max(float(np.median(steady)), 0.01)
        )
        save_params(out, seeds, runner[0].params)
        np.savez_compressed(
            out / "curves.npz",
            **{k: np.concatenate(v, axis=1) for k, v in curves.items()},  # ty: ignore[invalid-argument-type]
        )
        record["train_seconds"] = round(time.time() - t_start, 1)
        (out / "train.json").write_text(json.dumps(record, indent=1) + "\n")
        ret = metrics["return"][:, -1]
        print(
            f"update {done}/{cfg.num_updates}, {done * cfg.batch_steps:.3g} steps a seed, "
            f"{dt:.1f} s; return by seed {np.round(ret, 2).tolist()}",
            flush=True,
        )
        if deadline and done < cfg.num_updates and time.time() - T0 + 1.5 * max(steady) > deadline:
            record["stopped_early"] = f"deadline of {deadline:.0f} s after import"
            break
    record["first_chunk_seconds"] = record["chunks"][0]  # includes compilation
    stats = jax.devices()[0].memory_stats() or {}
    record["peak_device_bytes"] = stats.get("peak_bytes_in_use")
    record["train_seconds"] = round(time.time() - t_start, 1)
    (out / "train.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# ---------------------------------------------------------------------------- evaluation


def load_runs(runs: list[str]) -> tuple[list[dict], list]:
    """Each policy's record (run, seed, α) and parameters, from run folders or from
    `hf:<path>` folders of the private dataset."""
    from flax import serialization

    policies, params = [], []
    for run in runs:
        folder = Path(run)
        if run.startswith("hf:"):
            import os

            from huggingface_hub import snapshot_download

            sub = run[3:].strip("/")
            local = snapshot_download(
                "arclabs561/games-runs",
                repo_type="dataset",
                allow_patterns=[f"{sub}/*"],
                token=os.environ.get("HF_TOKEN"),
            )
            folder = Path(local) / sub
        record = json.loads((folder / "train.json").read_text())
        cfg = Config(**record["config"])
        for s in record["seeds"]:
            raw = (folder / f"params-seed{s}.msgpack").read_bytes()
            params.append((cfg, serialization.msgpack_restore(raw)))
            policies.append({"run": run, "seed": s, "ent_coef": cfg.ent_coef})
    return policies, params


def play_pairs(cfg: Config, stacked, pairs: np.ndarray, decks: np.ndarray, greedy: bool, seed: int):
    """Play every deal with every pair (i, j): policy i in seat 0, policy j in seat 1.
    Returns, per pair and deal, the score, the number of turns and the action of the player
    to move at each turn (-1 after the end)."""
    import jax
    import jax.numpy as jnp

    rules = H4_RULES
    env = make_env(rules)
    n_actions = env.num_moves
    network = build_network(n_actions, cfg)
    turns = max_turns(rules)

    def game(p0, p1, deck, key):
        obs, state = env.reset_from_deck_of_pairs(deck)
        carries = [zero_carry(1, cfg.hidden_dim), zero_carry(1, cfg.hidden_dim)]

        def turn(c, k):
            state, obs, carries = c
            legal = env.get_legal_moves(state)
            acts, new_carries = [], []
            for i, (p, agent) in enumerate(((p0, "agent_0"), (p1, "agent_1"))):
                avail = legal[agent][None]
                x = (obs[agent][None, None], jnp.zeros((1, 1), dtype=bool), avail[None])
                carry, logits, _ = network.apply(p, carries[i], x)
                logits = logits[0, 0]
                k, sub = jax.random.split(k)
                a = jnp.argmax(logits) if greedy else jax.random.categorical(sub, logits)
                acts.append(a)
                new_carries.append(carry)
            cur = jnp.argmax(state.cur_player_idx)
            obs2, state2, _, _, _ = env.step_env(k, state, {"agent_0": acts[0], "agent_1": acts[1]})
            over = state.done
            keep = lambda old, new: jax.tree.map(lambda x, y: jnp.where(over, x, y), old, new)  # noqa: E731
            act = jnp.where(over, -1, jnp.where(cur == 0, acts[0], acts[1]))
            return (keep(state, state2), keep(obs, obs2), keep(carries, new_carries)), act

        (final, _, _), acts = jax.lax.scan(
            turn, (state, obs, carries), jax.random.split(key, turns)
        )
        return final.score, (acts >= 0).sum(), acts, final.done

    def pair_games(i, j, keys):
        p0 = jax.tree.map(lambda x: x[i], stacked)
        p1 = jax.tree.map(lambda x: x[j], stacked)
        return jax.vmap(game, in_axes=(None, None, 0, 0))(p0, p1, jnp.asarray(decks), keys)

    stacked = jax.tree.map(jnp.asarray, stacked)
    run = jax.jit(jax.vmap(pair_games))
    key = jax.random.PRNGKey(seed)
    scores, lengths, actions = [], [], []
    block = 32 if jax.default_backend() == "gpu" else 8
    for b in range(0, len(pairs), block):
        ij = pairs[b : b + block]
        keys = jax.vmap(
            lambda i, j: jax.random.split(
                jax.random.fold_in(jax.random.fold_in(key, i), j), len(decks)
            )
        )(jnp.asarray(ij[:, 0]), jnp.asarray(ij[:, 1]))
        if len(ij) < block:  # pad to one shape, so that one program serves every block
            pad = block - len(ij)
            ij = np.concatenate([ij, np.repeat(ij[-1:], pad, axis=0)])
            keys = jnp.concatenate([keys, jnp.repeat(keys[-1:], pad, axis=0)])
        s, n, a, fin = jax.device_get(run(jnp.asarray(ij[:, 0]), jnp.asarray(ij[:, 1]), keys))
        assert np.all(fin), "a game outlasted the turn bound"
        k = min(block, len(pairs) - b)
        scores.append(s[:k])
        lengths.append(n[:k])
        actions.append(a[:k])
    return (
        np.concatenate(scores).astype(np.int8),
        np.concatenate(lengths).astype(np.int8),
        np.concatenate(actions).astype(np.int8),
    )


def evaluate(runs: list[str], deal_seeds: list[int], out: Path, sample_seed: int = 0) -> dict:
    """Every ordered pair of the runs' policies on every deal, greedy and sampled."""
    import jax

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    policies, loaded = load_runs(runs)
    cfgs = {c.hidden_dim for c, _ in loaded}
    assert len(cfgs) == 1, "policies of different sizes"
    cfg = loaded[0][0]
    stacked = jax.tree.map(lambda *xs: np.stack(xs), *[p for _, p in loaded])
    decks = np.stack([to_jaxmarl(H4_RULES, seeded_deck(H4_RULES, s)) for s in deal_seeds])
    n = len(policies)
    pairs = np.array([(i, j) for i in range(n) for j in range(n)], dtype=np.int32)
    record = {"policies": policies, "deal_seeds": [deal_seeds[0], deal_seeds[-1]], "modes": {}}
    arrays = {}
    for mode, greedy in (("greedy", True), ("sampled", False)):
        t = time.time()
        s, ln, a = play_pairs(cfg, stacked, pairs, decks, greedy, sample_seed)
        arrays[f"{mode}_scores"] = s.reshape(n, n, len(deal_seeds))
        arrays[f"{mode}_turns"] = ln.reshape(n, n, len(deal_seeds))
        arrays[f"{mode}_actions"] = a.reshape(n, n, len(deal_seeds), -1)
        record["modes"][mode] = {"seconds": round(time.time() - t, 1), "games": int(s.size)}
        print(f"{mode}: {s.size} games in {time.time() - t:.0f} s, mean {s.mean():.3f}", flush=True)
    np.savez_compressed(out / "games.npz", deal_seeds=np.array(deal_seeds), **arrays)
    record["seconds"] = round(time.time() - t0, 1)
    record["import_seconds"] = round(t0 - T0, 1)
    (out / "evaluate.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# ---------------------------------------------------------------------------- command line


def smoke(out: Path) -> None:
    """A tiny run of everything on a CPU: two seeds, a few updates, three deals."""
    from hanabi import h4

    cfg = Config(num_envs=8, num_steps=16, total_steps=8 * 16 * 6, fc_dim=16, hidden_dim=16)
    rec = train(cfg, [1, 2], out / "run", chunk=2)
    print("train:", rec["updates_done"], "updates,", rec["steps_per_second_all_seeds"], "steps/s")
    evaluate([str(out / "run")], [400000, 400001, 400002], out / "eval")
    h4.write_summary(out / "eval", [str(out / "run")])
    summary = json.loads((out / "eval" / "h4.json").read_text())
    print(json.dumps(summary["greedy"]["scores"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ent-coef", type=float, required=True)
    t.add_argument("--seeds", default="1-8")
    t.add_argument("--steps", type=float, default=2e8, help="environment steps per seed")
    t.add_argument("--num-envs", type=int, default=1024)
    t.add_argument("--chunk", type=int, default=20, help="updates between checkpoints")
    t.add_argument("--deadline-minutes", type=float, default=0)
    t.add_argument("--agreement", type=int, default=0, help="first run the agreement test, n games")
    t.add_argument("--evaluate-deals", default="", help="then evaluate on these deals (the pilot)")
    t.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--deals", default="400000-401999")
    e.add_argument("--sample-seed", type=int, default=0)
    e.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--out", type=Path, default=Path("/tmp/h4-smoke"))
    args = parser.parse_args()
    if args.cmd == "train":
        args.out.mkdir(parents=True, exist_ok=True)
        if args.agreement:
            from hanabi.jax_bridge import agreement_test

            report = agreement_test(args.agreement)
            (args.out / "agreement.json").write_text(json.dumps(report, indent=1) + "\n")
            ok = all(report[k]["agree"] for k in report if isinstance(report[k], dict))
            print("agreement test:", "passed" if ok else "FAILED", flush=True)
            if not ok:
                raise SystemExit("the engines disagree; plan H4 stops here")
        cfg = Config(ent_coef=args.ent_coef, total_steps=args.steps, num_envs=args.num_envs)
        rec = train(cfg, parse_seeds(args.seeds), args.out, args.chunk, args.deadline_minutes * 60)
        print(json.dumps({k: v for k, v in rec.items() if k != "chunks"}, indent=1))
        if args.evaluate_deals:
            from hanabi import h4

            evaluate([str(args.out)], parse_seeds(args.evaluate_deals), args.out / "eval")
            h4.write_summary(args.out / "eval", [str(args.out)])
    elif args.cmd == "evaluate":
        from hanabi import h4

        evaluate(args.runs, parse_seeds(args.deals), args.out, args.sample_seed)
        h4.write_summary(args.out, args.runs)
    else:
        smoke(args.out)
    print(f"total {time.time() - T0:.0f} s after import", flush=True)


if __name__ == "__main__":
    main()
