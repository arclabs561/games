"""Hashes of every computation in maps that feeds a quoted number, for comparison across CPUs."""
import hashlib, json, sys, time
import numpy as np
from maps import colors, data, shelf, sne, som, study
try:
    from maps import portable
except ImportError:  # main before the fix
    portable = None

def h(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float)).tobytes()).hexdigest()[:16]

full = "--full" in sys.argv
out = {}
t0 = time.time()
rng = np.random.default_rng(0)
x = rng.normal(size=100_000) * 30
out["numpy exp log cbrt pow atan2"] = h(np.concatenate([np.exp(x), np.log(np.abs(x)), np.cbrt(x), np.abs(x) ** 2.4, np.arctan2(x, x[::-1])]))
out["numpy matmul svd"] = h(np.concatenate([(x[:40000].reshape(200, 200) @ x[40000:80000].reshape(200, 200)).ravel(), np.linalg.svd(x[:6000].reshape(1000, 6), full_matrices=False)[2].ravel()]))
if portable is not None:
  out["portable.atan2"] = h(portable.atan2(x, x[::-1]))
  out["portable.exp"] = h(portable.exp(x))
  out["portable.log"] = h(portable.log(np.abs(x) + 1e-300))
  out["portable.cbrt"] = h(portable.cbrt(x))
  out["portable.power"] = h(portable.power(np.abs(x), 2.4))
_, css = data.css_shelf()
out["oklab css"] = h(colors.oklab(css))
out["oklch css"] = h(colors.oklch(css))
for name, pts in study.point_sets().items():
    out[f"P {name}"] = h(sne.conditional_p(sne.sq_distances(pts), 30.0)[0])
    out[f"pca {name}"] = h(study.pca(pts))
out["css orders"] = hashlib.sha256(json.dumps(study.css_orders()).encode()).hexdigest()[:16]
out["benchmark"] = hashlib.sha256(json.dumps(study.benchmark_scores(), sort_keys=True).encode()).hexdigest()[:16]
out["benchmark section"] = study.rounded(study.benchmark_section())["1D t-SNE"]
cm, _ = study.color_map(); out["som colors"] = h(cm.weights)
pm, _ = study.pokemon_map(); out["som pokemon"] = h(pm.weights)
out["som section"] = hashlib.sha256(json.dumps(study.section("som")).encode()).hexdigest()[:16]
# sne.ipynb: finite differences and the clusters
r = np.random.default_rng(0)
xx, yy = r.normal(size=(30, 4)), r.normal(size=(30, 2))
pc = sne.conditional_p(sne.sq_distances(xx), 8.0)[0]
for kind in ("sne", "ssne", "tsne"):
    tgt = pc if kind == "sne" else sne.joint_p(pc)
    c, g = sne.cost_grad(kind, yy, tgt)
    out[f"cost_grad {kind}"] = h(np.append(g.ravel(), c))
r = np.random.default_rng(3)
blobs = np.concatenate([c + r.normal(scale=1.0, size=(20, 11)) for c in np.eye(11) * 5.0])
for kind in ("sne", "ssne", "tsne"):
    out[f"blobs {kind}"] = h(sne.embed(sne.sq_distances(blobs), kind, perplexity=15, seed=0).y)
# scikit-learn comparisons printed by sne.ipynb (third-party code: BLAS and libm inside)
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence
_, stats, _ = data.pokemon()
d2 = sne.sq_distances(stats)
P = sne.joint_p(sne.conditional_p(d2, 30.0)[0])
P_sk = squareform(_joint_probabilities(d2, 30.0, 0))
y0 = np.random.default_rng(1).normal(size=(len(stats), 2))
cost, grad = sne.cost_grad("tsne", y0, P)
cost_sk, grad_sk = _kl_divergence(y0.ravel(), squareform(P, checks=False), 1, len(stats), 2)
out["sklearn P diff"] = f"{np.abs(P - P_sk).max():.1e}"
out["sklearn cost"] = f"{cost:.10f} vs {cost_sk:.10f}"
out["sklearn grad diff"] = f"{np.abs(grad.ravel() - grad_sk).max():.1e}"
out["seconds quick"] = round(time.time() - t0)
if full:
    for name in ("pokemon", "words"):
        for label, y in study.embeddings(name).items():
            out[f"map {name} {label}"] = h(y)
    for name in study.SECTIONS:
        out[f"section {name}"] = hashlib.sha256(json.dumps(study.section(name)).encode()).hexdigest()[:16]
    out["tsne seeds"] = study.rounded(study.tsne_seed_excess())
    from sklearn.manifold import TSNE
    sk = TSNE(perplexity=30, method="exact", init="random", random_state=0, learning_rate=100, early_exaggeration=4, max_iter=1000)
    y_sk = sk.fit_transform(stats)
    out["sklearn TSNE"] = f"{sk.kl_divergence_:.2f}"
    out["seconds"] = round(time.time() - t0)
for k, v in out.items():
    print(f"{k:24} {v}")
