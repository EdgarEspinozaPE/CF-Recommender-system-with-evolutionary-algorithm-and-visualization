"""
Convergencia del modelo adoptado respecto de su propia función de fitness.

Uso: python diagnostico_convergencia.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py. Salida: diagnostico_convergencia.json.
"""
import gzip
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import corridas_poblacion as cp
import explanation_metrics as em

POB = HERE / "poblacion"
GUARDADOS = HERE / "modelos"
SEMILLAS = list(range(42, 47))
LAMBDA = 0.001


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, semilla):
    ruta = POB / conjunto / "part42" / "modelos" / f"evo_dir_s0.3_seed{semilla}.json.gz"
    d = leer(ruta) if ruta.exists() else json.loads(
        (GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))
    return {"P": np.array(d["user_factors"]), "Q": np.array(d["item_factors"]),
            "bu": np.array(d["user_bias"]), "bi": np.array(d["item_bias"]),
            "mu": d["global_mean"], "fitness": d.get("fitness")}


def fitness(m, U, I, R):
    e = R - (m["mu"] + m["bu"][U] + m["bi"][I] + np.sum(m["P"][U] * m["Q"][I], axis=1))
    mse = float(np.mean(e * e))
    reg_f = float(np.sum(m["P"] ** 2) + np.sum(m["Q"] ** 2))
    reg_b = float(np.sum(m["bu"] ** 2) + np.sum(m["bi"] ** 2))
    return mse + LAMBDA * (reg_f + reg_b), mse, LAMBDA * reg_f, LAMBDA * reg_b, e


def gradiente(m, U, I, e):
    N = len(e)
    gP = 2 * LAMBDA * m["P"]
    gQ = 2 * LAMBDA * m["Q"]
    np.add.at(gP, U, -(2 / N) * (e[:, None] * m["Q"][I]))
    np.add.at(gQ, I, -(2 / N) * (e[:, None] * m["P"][U]))
    gu = 2 * LAMBDA * m["bu"]
    gi = 2 * LAMBDA * m["bi"]
    np.add.at(gu, U, -(2 / N) * e)
    np.add.at(gi, I, -(2 / N) * e)
    return {"P": gP, "Q": gQ, "bu": gu, "bi": gi}


def continuar(m0, U, I, R, iteraciones):
    """Descenso de gradiente de lote completo con busqueda lineal sobre F."""
    m = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m0.items()}
    F, _, _, _, e = fitness(m, U, I, R)
    paso = 50.0
    traza = []
    for t in range(iteraciones):
        g = gradiente(m, U, I, e)
        norma2 = sum(float(np.sum(g[k] ** 2)) for k in g)
        if t in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, iteraciones - 1):
            traza.append({"iter": t, "F": F, "norma_grad": norma2 ** 0.5,
                          "norma_PQ": pq_norma(m)})
        while True:
            c = dict(m)
            for k in g:
                c[k] = m[k] - paso * g[k]
            Fc, _, _, _, ec = fitness(c, U, I, R)
            if Fc <= F - 1e-4 * paso * norma2 or paso < 1e-6:
                break
            paso *= 0.5
        m, F, e = c, Fc, ec
        paso *= 1.25
    return m, traza


def pq_norma(m):
    """||PQ^T||_F sin construir la matriz."""
    return float(np.sqrt(np.sum((m["P"].T @ m["P"]) * (m["Q"].T @ m["Q"]))))


def dif_pq(a, b):
    """||PaQa^T - PbQb^T||_F / ||PaQa^T||_F, invariante a cualquier rotacion."""
    aa = np.sum((a["P"].T @ a["P"]) * (a["Q"].T @ a["Q"]))
    bb = np.sum((b["P"].T @ b["P"]) * (b["Q"].T @ b["Q"]))
    ab = np.sum((a["P"].T @ b["P"]) * (a["Q"].T @ b["Q"]))
    return float(np.sqrt(max(aa + bb - 2 * ab, 0.0)) / np.sqrt(aa)), float(ab / np.sqrt(aa * bb))


def sigma_max(U, I, valores, n_u, n_i, iteraciones=200, semilla=0):
    """Mayor valor singular de la matriz dispersa (U, I, valores), por iteracion de potencias."""
    v = np.random.default_rng(semilla).normal(size=n_i)
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iteraciones):
        u = np.bincount(U, weights=valores * v[I], minlength=n_u)
        s = np.linalg.norm(u)
        u /= s
        v = np.bincount(I, weights=valores * u[U], minlength=n_i)
        v /= np.linalg.norm(v)
    return float(s)


def sesgos_optimos(U, I, R, mu, n_u, n_i, N, iteraciones=100):
    """Sesgos que minimizan F con P = Q = 0: cresta alterna con penalizacion lambda*N."""
    bu, bi = np.zeros(n_u), np.zeros(n_i)
    cu, ci = np.bincount(U, minlength=n_u), np.bincount(I, minlength=n_i)
    for _ in range(iteraciones):
        bu = np.bincount(U, weights=R - mu - bi[I], minlength=n_u) / (cu + LAMBDA * N)
        bi = np.bincount(I, weights=R - mu - bu[U], minlength=n_i) / (ci + LAMBDA * N)
    return bu, bi


def analizar(conjunto, iteraciones):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    tr = np.array(split.train)
    U, I, R = tr[:, 0].astype(int), tr[:, 1].astype(int), tr[:, 2].astype(float)
    te = np.array(split.test)
    Ut, It, Rt = te[:, 0].astype(int), te[:, 1].astype(int), te[:, 2].astype(float)
    N = len(R)
    n_u, n_i = len(ds.idx_to_user), len(ds.idx_to_item)
    modelos = {s: parametros(conjunto, s) for s in SEMILLAS}

    filas = []
    for s, m in modelos.items():
        F, mse, rf, rb, e = fitness(m, U, I, R)
        filas.append({"semilla": s, "F": F, "mse": mse, "reg_factores": rf, "reg_sesgos": rb,
                      "F_guardado": m["fitness"], "norma_PQ": pq_norma(m),
                      "desv_P": float(m["P"].std()), "desv_Q": float(m["Q"].std()),
                      "sigma_residuo_modelo": sigma_max(U, I, e, n_u, n_i)})

    pares = []
    for a, b in itertools.combinations(SEMILLAS, 2):
        d, c = dif_pq(modelos[a], modelos[b])
        pares.append({"a": a, "b": b, "dif_relativa_PQ": d, "coseno_PQ": c})

    mu = modelos[42]["mu"]
    bu0, bi0 = sesgos_optimos(U, I, R, mu, n_u, n_i, N)
    cero = {"P": np.zeros_like(modelos[42]["P"]), "Q": np.zeros_like(modelos[42]["Q"]),
            "bu": bu0, "bi": bi0, "mu": mu}
    F0, mse0, _, rb0, e0 = fitness(cero, U, I, R)
    sigma0 = sigma_max(U, I, e0, n_u, n_i)
    rmse0 = float(np.sqrt(np.mean((Rt - (mu + bu0[Ut] + bi0[It])) ** 2)))

    continuaciones = []
    for s in (42, 43):
        m, traza = continuar(modelos[s], U, I, R, iteraciones)
        Ff, msef, rff, rbf, _ = fitness(m, U, I, R)
        rmse = float(np.sqrt(np.mean((Rt - (m["mu"] + m["bu"][Ut] + m["bi"][It]
                                            + np.sum(m["P"][Ut] * m["Q"][It], axis=1))) ** 2)))
        continuaciones.append({"semilla": s, "F_final": Ff, "mse": msef, "reg_factores": rff,
                               "reg_sesgos": rbf, "norma_PQ": pq_norma(m),
                               "desv_P": float(m["P"].std()), "desv_Q": float(m["Q"].std()),
                               "rmse_prueba": rmse, "traza": traza})
        print(f"[{conjunto}] continuacion semilla {s}: F {filas[s - 42]['F']:.4f} -> {Ff:.4f}, "
              f"||PQ|| {filas[s - 42]['norma_PQ']:.2f} -> {pq_norma(m):.4f}", flush=True)

    return {"N": N, "lambda_por_N": LAMBDA * N, "modelos": filas, "pares_PQ": pares,
            "dif_relativa_PQ_media": statistics.fmean(p["dif_relativa_PQ"] for p in pares),
            "coseno_PQ_medio": statistics.fmean(p["coseno_PQ"] for p in pares),
            "factores_nulos": {"F": F0, "mse": mse0, "reg_sesgos": rb0,
                               "sigma_max_residuo": sigma0, "rmse_prueba": rmse0},
            "continuacion": continuaciones}


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"]
    ruta = HERE / "diagnostico_convergencia.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        total[c] = analizar(c, 1000)
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{c}: guardado", flush=True)
