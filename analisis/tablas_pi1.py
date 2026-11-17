"""
Tablas de PI-1: ablación del operador de mutación.

Uso: python tablas_pi1.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py y estabilidad_poblacion.json.
Salida: tablas_pi1.json.
"""
import gzip
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import corridas_poblacion as cp
import explanation_metrics as em

POB = HERE / "poblacion"
GUARDADOS = HERE / "modelos"
VECINOS = 10
CONFIGURACIONES = [
    "evo_nodir_s0.4082", "evo_dir_s0.1", "evo_dir_s0.3", "evo_dir_s0.4082",
    "sgd_lr0.02_ep10_s0.1", "sgd_lr0.005_ep30_s0.1", "sgd_lr0.005_ep10_s0.1",
]
METRICAS = ["rmse", "mae", "p10", "ndcg10", "confidence", "preference_match",
            "ablation_shift", "necessity"]


def leer_gz(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, modelo):
    ruta = POB / conjunto / "part42" / "modelos" / f"{modelo}.json.gz"
    if ruta.exists():
        return leer_gz(ruta)
    return json.loads((GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))


def solapamiento(listas):
    n = len(listas)
    cuenta = Counter(i for lista in listas for i in lista)
    return sum(c * (c - 1) for c in cuenta.values()) / (n * (n - 1))


def cuota_latente(d):
    pu, qi = np.array(d["user_factors"]), np.array(d["item_factors"])
    bu, bi = np.array(d["user_bias"]), np.array(d["item_bias"])
    paso = max(1, len(qi) // 240)
    latente = pu @ qi[::paso].T
    score = d["global_mean"] + bu[:, None] + bi[::paso][None, :] + latente
    return float(np.mean(latente.var(axis=1) / np.maximum(score.var(axis=1), 1e-12)))


def vecinos(matriz):
    x = matriz / np.maximum(np.linalg.norm(matriz, axis=1, keepdims=True), 1e-12)
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1)[:, :VECINOS]


def jaccard_vecinos(a, b):
    return statistics.fmean(len(set(x) & set(y)) / len(set(x) | set(y))
                            for x, y in zip(a.tolist(), b.tolist()))


def azar_vecinos(n):
    esperado = VECINOS * VECINOS / (n - 1)
    return esperado / (2 * VECINOS - esperado)


# TABLA DE ABLACIÓN
def tabla(conjunto, estabilidad):
    carpeta = POB / conjunto / "part42"
    sesgos = leer_gz(carpeta / "sesgos.json.gz")
    out = {"sesgos": {**sesgos["resumen"], "solapamiento": solapamiento(sesgos["recs"])},
           "configuraciones": {}}
    for cfg in CONFIGURACIONES:
        rutas = sorted(carpeta.glob(f"{cfg}_seed*.json.gz"))
        if not rutas:
            continue
        res = [leer_gz(r) for r in rutas]
        semillas = [int(re.search(r"seed(\d+)", r.name).group(1)) for r in rutas]
        fila = {"semillas": semillas}
        for m in METRICAS:
            valores = [r["resumen"][m] for r in res]
            fila[m] = {"media": statistics.fmean(valores), "min": min(valores), "max": max(valores)}
        fila["solapamiento"] = statistics.fmean(solapamiento(r["recs"]) for r in res)
        fila["sigma_lat"] = statistics.fmean(statistics.fmean(r["lat_sd"]) for r in res)
        e = estabilidad[conjunto]["configuraciones"][cfg]["resumen"]
        fila["J_rec"] = e["J_rec"]["media"]
        fila["J_vec"] = e["J_vecinos_items"]["media"]
        modelos = [parametros(conjunto, f"{cfg}_seed{s}") for s in semillas]
        fila["desviacion_factores"] = statistics.fmean(
            float(np.std(np.vstack([np.array(d["user_factors"]), np.array(d["item_factors"])])))
            for d in modelos)
        if cfg.startswith("evo"):
            fila["cuota_varianza_latente"] = statistics.fmean(cuota_latente(d) for d in modelos[:2])
        out["configuraciones"][cfg] = fila
    a = leer_gz(carpeta / "evo_nodir_s0.4082_seed42.json.gz")["recs"]
    b = leer_gz(carpeta / "evo_nodir_s0.4082_seed43.json.gz")["recs"]
    out["no_dirigido_items_comunes_top10"] = statistics.fmean(
        len(set(x) & set(y)) for x, y in zip(a, b))
    return out


# VECINDARIOS DE LOS ÍTEMS MÁS VALORADOS
def vecinos_populares(conjunto, n_items=100):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    cuenta = Counter(i for _, i, _ in split.train)
    seleccion = np.array([i for i, _ in cuenta.most_common(n_items)])
    pares = [(a, b) for a in range(42, 47) for b in range(a + 1, 47)]
    q = {s: np.array(parametros(conjunto, f"evo_dir_s0.3_seed{s}")["item_factors"])[seleccion]
         for s in range(42, 47)}
    valores = [jaccard_vecinos(vecinos(q[a]), vecinos(q[b])) for a, b in pares]
    return {"items": n_items, "valoraciones_minimas": cuenta[int(seleccion[-1])],
            "J_vec_media": statistics.fmean(valores), "J_vec_azar": azar_vecinos(n_items)}


def main(conjuntos):
    estabilidad = json.loads((HERE / "estabilidad_poblacion.json").read_text(encoding="utf-8"))
    ruta = HERE / "tablas_pi1.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        total[c] = tabla(c, estabilidad)
        if c == "movielens":
            total[c]["vecinos_items_populares"] = vecinos_populares(c)
        s = total[c]["sesgos"]
        print(f"[{c}] solo sesgos: RMSE {s['rmse']:.4f}  NDCG@10 {s['ndcg10']:.4f}")
        for cfg, f in total[c]["configuraciones"].items():
            print(f"[{c}] {cfg:<24} RMSE {f['rmse']['media']:.4f}  NDCG@10 {f['ndcg10']['media']:.4f}  "
                  f"solap. {f['solapamiento']:.2f}  sigma_lat {f['sigma_lat']:.3f}  "
                  f"J_rec {f['J_rec']:.3f}  J_vec {f['J_vec']:.4f}")
    ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"])
