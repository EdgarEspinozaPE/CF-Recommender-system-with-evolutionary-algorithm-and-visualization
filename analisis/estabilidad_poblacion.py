"""
Estabilidad entre semillas: listas, factores dominantes, evidencia y vecindarios
latentes, con su nivel de azar.

Uso: python estabilidad_poblacion.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py. Salida: estabilidad_poblacion.json.
"""
import gzip
import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path

import os

import numpy as np

HERE = Path(__file__).resolve().parent
POB = Path(os.environ.get("POB_SALIDA", HERE / "poblacion"))
GUARDADOS = HERE / "modelos"
SALIDA = Path(os.environ.get("EST_SALIDA", HERE / "estabilidad_poblacion.json"))
K, K_DOM, K_VECINOS = 6, 3, 10


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, id_modelo):
    ruta = POB / conjunto / "part42" / "modelos" / f"{id_modelo}.json.gz"
    if ruta.exists():
        d = leer(ruta)
    elif id_modelo == "evo_dir_s0.3_seed42":
        d = json.loads((GUARDADOS / f"{conjunto}_seed42_train.json")
                       .read_text(encoding="utf-8"))
    else:
        return None
    return np.array(d["user_factors"]), np.array(d["item_factors"])


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def azar_jaccard(k, kd):
    """Jaccard esperado entre dos subconjuntos de kd elementos de k, al azar."""
    total = math.comb(k, kd)
    return sum(math.comb(kd, c) * math.comb(k - kd, kd - c) / total * c / (2 * kd - c)
               for c in range(kd + 1))


def procrustes(Pa, Qa, Pb, Qb):
    """Rotacion ortogonal R que minimiza ||[Pb; Qb] R - [Pa; Qa]||_F."""
    Xa, Xb = np.vstack([Pa, Qa]), np.vstack([Pb, Qb])
    U, _, Vt = np.linalg.svd(Xb.T @ Xa)
    R = U @ Vt
    norma = np.linalg.norm(Xa)
    return R, float(np.linalg.norm(Xb @ R - Xa) / norma), float(np.linalg.norm(Xb - Xa) / norma)


def dominantes(p, q):
    return np.argsort(-np.abs(p * q), kind="stable")[:K_DOM].tolist()


def vecinos(M, bloque=1000):
    norma = np.linalg.norm(M, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    X = M / norma
    out = np.empty((len(M), K_VECINOS), dtype=np.int64)
    for s in range(0, len(M), bloque):
        S = X[s:s + bloque] @ X.T
        filas = np.arange(S.shape[0])
        S[filas, s + filas] = -np.inf
        out[s:s + S.shape[0]] = np.argpartition(-S, K_VECINOS, axis=1)[:, :K_VECINOS]
    return out


def jaccard_vecinos(Na, Nb):
    return statistics.fmean(jaccard(a, b) for a, b in zip(Na.tolist(), Nb.tolist()))


def cargar_resultado(conjunto, id_modelo):
    r = leer(POB / conjunto / "part42" / f"{id_modelo}.json.gz")
    f = r["filas"]
    por_par = {(u, i): (fa, ev) for u, i, fa, ev in
               zip(f["usuario"], f["item"], f["factores"], f["evidencia_items"])}
    return r["recs"], por_par


def resumir(valores):
    if not valores:
        return None
    return {"media": statistics.fmean(valores),
            "desv": statistics.stdev(valores) if len(valores) > 1 else 0.0,
            "min": min(valores), "max": max(valores), "n_pares": len(valores)}


def analizar(conjunto):
    ctx = leer(POB / conjunto / "part42" / "contexto.json.gz")
    poblacion = ctx["poblacion"]
    configs = {}
    for ruta in sorted((POB / conjunto / "part42").glob("*.json.gz")):
        m = re.fullmatch(r"(.+)_seed(\d+)\.json\.gz", ruta.name)
        if m:
            configs.setdefault(m.group(1), []).append(int(m.group(2)))

    salida = {"azar_J_fac": azar_jaccard(K, K_DOM), "configuraciones": {}}
    for config, semillas in sorted(configs.items()):
        semillas.sort()
        if len(semillas) < 2:
            continue
        ids = {s: f"{config}_seed{s}" for s in semillas}
        res = {s: cargar_resultado(conjunto, ids[s]) for s in semillas}
        params = {s: parametros(conjunto, ids[s]) for s in semillas}
        vec = {s: (vecinos(params[s][1]), vecinos(params[s][0]))
               for s in semillas if params[s] is not None}
        pares = []
        for a, b in itertools.combinations(semillas, 2):
            recs_a, filas_a = res[a]
            recs_b, filas_b = res[b]
            j_rec = statistics.fmean(jaccard(x, y) for x, y in zip(recs_a, recs_b))
            comunes = [k for k in filas_a if k in filas_b]
            j_fac = [jaccard(filas_a[k][0], filas_b[k][0]) for k in comunes]
            j_evid = [jaccard(filas_a[k][1], filas_b[k][1]) for k in comunes
                      if filas_a[k][1] or filas_b[k][1]]
            par = {"a": a, "b": b, "J_rec": j_rec, "recomendaciones_comunes": len(comunes),
                   "J_fac": statistics.fmean(j_fac) if j_fac else None,
                   "J_evid": statistics.fmean(j_evid) if j_evid else None}
            if params[a] is not None and params[b] is not None:
                (Pa, Qa), (Pb, Qb) = params[a], params[b]
                R, err, err0 = procrustes(Pa, Qa, Pb, Qb)
                PbR, QbR = Pb @ R, Qb @ R
                discrepancias = sum(sorted(dominantes(Pa[u], Qa[i])) != sorted(filas_a[(u, i)][0])
                                    for u, i in comunes)
                j_al = [jaccard(filas_a[(u, i)][0], dominantes(PbR[u], QbR[i]))
                        for u, i in comunes]
                par.update({
                    "J_fac_alineado": statistics.fmean(j_al) if j_al else None,
                    "error_relativo_sin_alinear": err0, "error_relativo_alineado": err,
                    "J_vecinos_items": jaccard_vecinos(vec[a][0], vec[b][0]),
                    "J_vecinos_usuarios": jaccard_vecinos(vec[a][1], vec[b][1]),
                    "control_firmas_distintas": discrepancias})
            pares.append(par)
            print(f"  {conjunto} {config} {a}-{b}: " + " ".join(
                f"{k}={v:.4f}" for k, v in par.items()
                if isinstance(v, float)), flush=True)
        claves = ["J_rec", "J_fac", "J_fac_alineado", "J_evid", "J_vecinos_items",
                  "J_vecinos_usuarios", "error_relativo_sin_alinear",
                  "error_relativo_alineado"]
        salida["configuraciones"][config] = {
            "pares": pares,
            "resumen": {k: resumir([p[k] for p in pares if p.get(k) is not None])
                        for k in claves}}
    return salida


if __name__ == "__main__":
    total = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {}
    for conjunto in sys.argv[1:] or ["movielens", "lastfm", "bookcrossing"]:
        if not (POB / conjunto / "part42" / "contexto.json.gz").exists():
            print(f"{conjunto}: sin resultados todavia")
            continue
        total[conjunto] = analizar(conjunto)
        SALIDA.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{conjunto}: guardado en {SALIDA.name}", flush=True)
