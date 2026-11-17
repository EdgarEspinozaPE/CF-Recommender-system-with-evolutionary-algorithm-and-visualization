"""
Rejilla del descenso de gradiente (tasa, épocas y escala) con las semillas 42 y 43.

Uso: python barrido_sgd_poblacion.py [conjunto ...]    (por defecto, movielens)
Salida: barrido_sgd_poblacion.json.
"""
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import corridas_poblacion as cp
import explanation_metrics as em
from baseline_sgd import SGDMatrixFactorization

POB = HERE / "poblacion"
TOP_N = 10


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def listas(P, Q, bu, bi, mu, usuarios, conocidos):
    salida = []
    for u in usuarios:
        s = mu + bu[u] + bi + Q @ P[u]
        vistos = list(conocidos.get(u, ()))
        if vistos:
            s[vistos] = -np.inf
        top = np.argpartition(-s, TOP_N)[:TOP_N]
        salida.append([int(i) for i in top[np.argsort(-s[top], kind="stable")]])
    return salida


def evaluar(modelo, ctx, test, conocidos):
    b = modelo.best_
    P, Q = np.array(b.user_factors), np.array(b.item_factors)
    bu, bi = np.array(b.user_bias), np.array(b.item_bias)
    mu = b.global_mean
    U, I, R = test
    pred = np.clip(mu + bu[U] + bi[I] + np.sum(P[U] * Q[I], axis=1), 1.0, 5.0)
    e = R - pred
    recs = listas(P, Q, bu, bi, mu, ctx["poblacion"], conocidos)
    pos = {u: k for k, u in enumerate(ctx["poblacion"])}
    p10, r10, ndcg, hit = [], [], [], []
    for u in ctx["ranking"]:
        lista, rel = recs[pos[u]], ctx["relevantes"].get(u, set())
        aciertos = len(set(lista) & rel)
        p10.append(aciertos / TOP_N)
        r10.append(aciertos / len(rel))
        ndcg.append(em.ndcg_at_k(lista, rel, TOP_N))
        hit.append(1 if aciertos else 0)
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "p10": statistics.fmean(p10), "r10": statistics.fmean(r10),
            "ndcg10": statistics.fmean(ndcg), "hit10": statistics.fmean(hit)}, recs, P, Q


def coseno_pq(Pa, Qa, Pb, Qb):
    aa = np.sum((Pa.T @ Pa) * (Qa.T @ Qa))
    bb = np.sum((Pb.T @ Pb) * (Qb.T @ Qb))
    ab = np.sum((Pa.T @ Pb) * (Qa.T @ Qb))
    return float(ab / np.sqrt(aa * bb))


def barrer(conjunto, ruta, total):
    ctx = cp.contexto(conjunto, 42)
    ds, split = ctx["ds"], ctx["split"]
    conocidos = ctx["known_map"]
    te = np.array(split.test)
    test = (te[:, 0].astype(int), te[:, 1].astype(int), te[:, 2].astype(float))
    filas = total.setdefault(conjunto, [])
    hechas = {(f["lr"], f["epocas"], f["escala"]) for f in filas}
    for lr in (0.005, 0.01, 0.02):
        for epocas in (10, 30, 60):
            for escala in (0.1, 0.3):
                if (lr, epocas, escala) in hechas:
                    continue
                res, recs, fact, seg = [], [], [], []
                for semilla in (42, 43):
                    t = time.time()
                    m = SGDMatrixFactorization(
                        n_users=len(ds.idx_to_user), n_items=len(ds.idx_to_item),
                        learning_rate=lr, epochs=epocas, init_scale=escala,
                        seed=semilla).fit(split.train)
                    seg.append(time.time() - t)
                    q, r, P, Q = evaluar(m, ctx, test, conocidos)
                    res.append(q)
                    recs.append(r)
                    fact.append((P, Q))
                fila = {"lr": lr, "epocas": epocas, "escala": escala,
                        "segundos": statistics.fmean(seg)}
                for k in res[0]:
                    fila[k] = statistics.fmean(x[k] for x in res)
                fila["J_rec"] = statistics.fmean(jaccard(a, b) for a, b in zip(*recs))
                fila["coseno_PQ"] = coseno_pq(*fact[0], *fact[1])
                guardado = POB / conjunto / "part42" / f"sgd_lr{lr}_ep{epocas}_s{escala}_seed42.json.gz"
                if guardado.exists():
                    ref = leer(guardado)
                    fila["coincide_listas_guardadas"] = statistics.fmean(
                        jaccard(a, b) for a, b in zip(recs[0], ref["recs"]))
                filas.append(fila)
                ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"[{conjunto}] lr={lr} ep={epocas} s={escala}: "
                      + ", ".join(f"{k}={v:.4f}" for k, v in fila.items()
                                  if k not in ("lr", "epocas", "escala")), flush=True)


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens"]
    ruta = Path(os.environ.get("BARRIDO_SALIDA", HERE / "barrido_sgd_poblacion.json"))
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        barrer(c, ruta, total)
