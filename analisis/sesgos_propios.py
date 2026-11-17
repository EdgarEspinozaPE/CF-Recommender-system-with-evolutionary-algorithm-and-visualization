"""
Origen de la coincidencia de listas entre semillas: Jaccard con las listas de
solo sesgos.

Uso: python sesgos_propios.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py. Salida: sesgos_propios.json.
"""
import gzip
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
CONFIGURACIONES = ["evo_dir_s0.1", "evo_dir_s0.4082", "evo_nodir_s0.4082",
                   "sgd_lr0.005_ep10_s0.1", "sgd_lr0.005_ep30_s0.1", "sgd_lr0.02_ep10_s0.1"]


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def media_jaccard(listas_a, listas_b):
    return statistics.fmean(jaccard(x, y) for x, y in zip(listas_a, listas_b))


def listas_top10(modelo, usuarios, conocidos, con_latente):
    P, Q = np.array(modelo["user_factors"]), np.array(modelo["item_factors"])
    bi, bu = np.array(modelo["item_bias"]), np.array(modelo["user_bias"])
    mu = modelo["global_mean"]
    salida = []
    for u in usuarios:
        s = mu + bu[u] + bi
        if con_latente:
            s = s + Q @ P[u]
        vistos = list(conocidos.get(u, ()))
        if vistos:
            s[vistos] = -np.inf
        top = np.argpartition(-s, 10)[:10]
        salida.append([int(i) for i in top[np.argsort(-s[top], kind="stable")]])
    return salida


def fraccion_comunes(listas_a, listas_b, referencia_a, referencia_b):
    comunes = tambien = 0
    for x, y, z, w in zip(listas_a, listas_b, referencia_a, referencia_b):
        c = set(x) & set(y)
        comunes += len(c)
        tambien += len(c & (set(z) | set(w)))
    return tambien / comunes if comunes else 0.0


def analizar(conjunto):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    conocidos = {}
    for u, i, _ in split.train:
        conocidos.setdefault(u, set()).add(i)
    base = POB / conjunto / "part42"
    usuarios = leer(base / "contexto.json.gz")["poblacion"]
    analiticos = leer(base / "sesgos.json.gz")["recs"]
    salida = {}
    for cfg in CONFIGURACIONES:
        r42 = leer(base / f"{cfg}_seed42.json.gz")["recs"]
        r43 = leer(base / f"{cfg}_seed43.json.gz")["recs"]
        m42 = leer(base / "modelos" / f"{cfg}_seed42.json.gz")
        m43 = leer(base / "modelos" / f"{cfg}_seed43.json.gz")
        reproduccion = media_jaccard(listas_top10(m42, usuarios, conocidos, True), r42)
        s42 = listas_top10(m42, usuarios, conocidos, False)
        s43 = listas_top10(m43, usuarios, conocidos, False)
        salida[cfg] = {
            "reproduccion_listas": reproduccion,
            "J_rec": media_jaccard(r42, r43),
            "J_sesgos_propios": statistics.fmean([media_jaccard(r42, s42), media_jaccard(r43, s43)]),
            "J_rec_solo_sesgos_propios": media_jaccard(s42, s43),
            "comunes_en_sesgos_propios": fraccion_comunes(r42, r43, s42, s43),
            "J_sesgos_analiticos": statistics.fmean([media_jaccard(r42, analiticos),
                                                     media_jaccard(r43, analiticos)]),
            "comunes_en_sesgos_analiticos": fraccion_comunes(r42, r43, analiticos, analiticos)}
        print(f"[{conjunto}] {cfg}: " + ", ".join(f"{k}={v:.4f}" for k, v in salida[cfg].items()),
              flush=True)
    return salida


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"]
    ruta = HERE / "sesgos_propios.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        total[c] = analizar(c)
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
