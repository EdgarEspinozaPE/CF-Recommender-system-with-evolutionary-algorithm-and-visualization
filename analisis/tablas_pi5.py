"""
Tablas de PI-5: comparación con el descenso de gradiente y barrido de gamma.

Uso: python tablas_pi5.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py y estabilidad_poblacion.json.
Salida: tablas_pi5.json.
"""
import gzip
import json
import random
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
CONFIGS = {"adoptado": "evo_dir_s0.3", "evo_s0.1": "evo_dir_s0.1",
           "sgd_lr0.02_ep10": "sgd_lr0.02_ep10_s0.1",
           "sgd_lr0.005_ep30": "sgd_lr0.005_ep30_s0.1",
           "sgd_lr0.005_ep10": "sgd_lr0.005_ep10_s0.1"}


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, id_modelo):
    ruta = POB / conjunto / "part42" / "modelos" / f"{id_modelo}.json.gz"
    if ruta.exists():
        return leer(ruta)
    return json.loads((GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))


def calidad(res, n_items):
    e = np.array(res["errores"])
    r = res["ranking"]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "p10": statistics.fmean(r["p10"]), "r10": statistics.fmean(r["r10"]),
            "ndcg10": statistics.fmean(r["ndcg10"]), "hit10": statistics.fmean(r["acierto"]),
            "cobertura_catalogo": len({i for lista in res["recs"] for i in lista}) / n_items}


def explicativas(f):
    return {k: statistics.fmean(f[c]) for k, c in [
        ("ablation_shift", "ablation_shift"), ("necesidad", "necessity"),
        ("suficiencia", "sufficiency"), ("cobertura_intrinseca", "intrinsic"),
        ("cobertura_evidencia", "evidencia"), ("confianza", "confidence"),
        ("preference_match", "preference_match")]}


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else None


def intra_usuario(grupos, a, b):
    valores = [pearson(a[g], b[g]) for g in grupos if len(g) >= 3]
    valores = [v for v in valores if v is not None]
    return statistics.fmean(valores)


def correlaciones(d, f):
    P, Q = np.array(d["user_factors"]), np.array(d["item_factors"])
    bu, bi, mu = np.array(d["user_bias"]), np.array(d["item_bias"]), d["global_mean"]
    U, I = np.array(f["usuario"]), np.array(f["item"])
    pm = np.array(f["preference_match"])
    lat = np.sum(P[U] * Q[I], axis=1)
    crudo = mu + bu[U] + bi[I] + lat
    orden = np.argsort(U, kind="stable")
    _, inicio = np.unique(U[orden], return_index=True)
    grupos = np.split(orden, inicio[1:])
    return {"corr_pm_score": intra_usuario(grupos, pm, crudo),
            "corr_pm_latente": intra_usuario(grupos, pm, lat)}


def fitness(d, U, I, R):
    P, Q = np.array(d["user_factors"]), np.array(d["item_factors"])
    bu, bi, mu = np.array(d["user_bias"]), np.array(d["item_bias"]), d["global_mean"]
    e = R - (mu + bu[U] + bi[I] + np.sum(P[U] * Q[I], axis=1))
    mse = float(np.mean(e * e))
    reg_f = LAMBDA * float(np.sum(P ** 2) + np.sum(Q ** 2))
    reg_b = LAMBDA * float(np.sum(bu ** 2) + np.sum(bi ** 2))
    return {"fitness": mse + reg_f + reg_b, "mse_entrenamiento": mse,
            "reg_factores": reg_f, "reg_sesgos": reg_b,
            "desv_P": float(P.std()), "desv_Q": float(Q.std())}


def media_y_desv(filas):
    salida = {}
    for k in filas[0]:
        v = [x[k] for x in filas if x[k] is not None]
        salida[k] = {"media": statistics.fmean(v),
                     "desv": statistics.stdev(v) if len(v) > 1 else 0.0,
                     "min": min(v), "max": max(v), "valores": v}
    return salida


def fila_modelo(conjunto, id_modelo, n_items, U, I, R):
    res = leer(POB / conjunto / "part42" / f"{id_modelo}.json.gz")
    d = parametros(conjunto, id_modelo)
    fila = {**calidad(res, n_items), **explicativas(res["filas"]),
            **correlaciones(d, res["filas"]), **fitness(d, U, I, R)}
    seg = res.get("segundos_entrenamiento")
    fila["segundos"] = seg
    return fila


def comparar(conjunto):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    n_items = len(ds.idx_to_item)
    tr = np.array(split.train)
    U, I, R = tr[:, 0].astype(int), tr[:, 1].astype(int), tr[:, 2].astype(float)
    est = json.loads((HERE / "estabilidad_poblacion.json").read_text(encoding="utf-8"))[conjunto]["configuraciones"]
    salida = {}
    for nombre, cfg in CONFIGS.items():
        semillas = SEMILLAS if nombre == "adoptado" else [42, 43]
        filas = [fila_modelo(conjunto, f"{cfg}_seed{s}", n_items, U, I, R) for s in semillas]
        salida[nombre] = {"metricas": media_y_desv(filas),
                          "estabilidad": est[cfg]["resumen"], "n_semillas": len(semillas)}
    salida["sesgos"] = {"metricas": calidad(leer(POB / conjunto / "part42" / "sesgos.json.gz"), n_items)}
    return salida


def usuarios_termino(semilla, train):
    """Reproduce la muestra del termino de explicabilidad de evolutionary_cf.fit."""
    azar = random.Random(semilla + 10_000)
    con_historial = sorted({u for u, _, _ in train})
    return azar.sample(con_historial, min(8, len(con_historial)))


def barrido_gamma():
    conjunto = "movielens"
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    n_items = len(ds.idx_to_item)
    tr = np.array(split.train)
    U, I, R = tr[:, 0].astype(int), tr[:, 1].astype(int), tr[:, 2].astype(float)
    est = json.loads((HERE / "estabilidad_poblacion.json").read_text(encoding="utf-8"))[conjunto]["configuraciones"]
    cfgs = {0.0: "evo_dir_s0.3", 0.5: "evo_dir_s0.3_gam0.5", 2.0: "evo_dir_s0.3_gam2.0"}
    por_gamma, muestra = {}, {}
    for g, cfg in cfgs.items():
        filas, pm_mu, pm_re = [], [], []
        for s in SEMILLAS:
            id_modelo = f"{cfg}_seed{s}"
            res = leer(POB / conjunto / "part42" / f"{id_modelo}.json.gz")
            filas.append(fila_modelo(conjunto, id_modelo, n_items, U, I, R))
            elegidos = set(usuarios_termino(s, split.train))
            f = res["filas"]
            pm = np.array(f["preference_match"])
            en = np.array([u in elegidos for u in f["usuario"]])
            pm_mu.append(float(pm[en].mean()))
            pm_re.append(float(pm[~en].mean()))
        por_gamma[str(g)] = {"metricas": media_y_desv(filas), "estabilidad": est[cfg]["resumen"]}
        muestra[str(g)] = {"pm_muestra": pm_mu, "pm_resto": pm_re}
    pareadas = {}
    base = por_gamma["0.0"]["metricas"]
    for g in ("0.5", "2.0"):
        otra = por_gamma[g]["metricas"]
        d = {}
        for k in base:
            a, b = base[k]["valores"], otra[k]["valores"]
            if k == "segundos" or len(a) != len(b):
                continue
            dif = [y - x for x, y in zip(a, b)]
            d[k] = {"diferencias": dif, "media": statistics.fmean(dif),
                    "relativa_media": statistics.fmean(dif) / abs(statistics.fmean(a)),
                    "positivas": sum(1 for x in dif if x > 0),
                    "negativas": sum(1 for x in dif if x < 0)}
        pareadas[g] = d
    return {"por_gamma": por_gamma, "pareadas": pareadas, "muestra": muestra}


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"]
    ruta = HERE / "tablas_pi5.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    total.setdefault("comparacion", {})
    for c in conjuntos:
        total["comparacion"][c] = comparar(c)
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{c}: guardado", flush=True)
    if "movielens" in conjuntos:
        total["gamma"] = barrido_gamma()
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print("gamma: guardado", flush=True)
