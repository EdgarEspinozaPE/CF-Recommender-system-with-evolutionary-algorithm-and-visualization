"""
Control de PI-5: una única trayectoria de gradiente equiparable al modelo adoptado.

Uso: python control_trayectoria.py <conjunto> [semilla ...]    entrena y evalúa
     python control_trayectoria.py resumen                     escribe el JSON

Salida: control_trayectoria.json.
"""
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
import estabilidad_poblacion as ep
import explanation_metrics as em
import tablas_pi5 as t5

PASOS = 120 * 8192
ETA, ESCALA = 0.01, 0.3
EPOCAS = {"movielens": 12, "bookcrossing": 10, "lastfm": 17}
SEMILLAS = list(range(42, 47))


def entrenar(conjunto, semillas):
    cp.plan = lambda _: [(42, cp.sgd(ETA, EPOCAS[conjunto], ESCALA, s)) for s in semillas]
    cp.correr(conjunto, ["trayectoria"])


def coseno_pq(a, b):
    Pa, Qa, Pb, Qb = a[0], a[1], b[0], b[1]
    aa = np.sum((Pa.T @ Pa) * (Qa.T @ Qa))
    bb = np.sum((Pb.T @ Pb) * (Qb.T @ Qb))
    ab = np.sum((Pa.T @ Pb) * (Qa.T @ Qb))
    return float(ab / np.sqrt(aa * bb))


def estabilidad(conjunto, config):
    ids = {s: f"{config}_seed{s}" for s in SEMILLAS}
    res = {s: ep.cargar_resultado(conjunto, ids[s]) for s in SEMILLAS}
    params = {s: ep.parametros(conjunto, ids[s]) for s in SEMILLAS}
    vec = {s: (ep.vecinos(params[s][1]), ep.vecinos(params[s][0])) for s in SEMILLAS}
    pares = []
    for a, b in itertools.combinations(SEMILLAS, 2):
        recs_a, filas_a = res[a]
        recs_b, filas_b = res[b]
        comunes = [k for k in filas_a if k in filas_b]
        (Pa, Qa), (Pb, Qb) = params[a], params[b]
        R, err, err0 = ep.procrustes(Pa, Qa, Pb, Qb)
        PbR, QbR = Pb @ R, Qb @ R
        j_evid = [ep.jaccard(filas_a[k][1], filas_b[k][1]) for k in comunes
                  if filas_a[k][1] or filas_b[k][1]]
        pares.append({
            "J_rec": statistics.fmean(ep.jaccard(x, y) for x, y in zip(recs_a, recs_b)),
            "J_evid": statistics.fmean(j_evid) if j_evid else None,
            "J_fac_alineado": statistics.fmean(
                ep.jaccard(filas_a[(u, i)][0], ep.dominantes(PbR[u], QbR[i])) for u, i in comunes),
            "J_vecinos_items": ep.jaccard_vecinos(vec[a][0], vec[b][0]),
            "J_vecinos_usuarios": ep.jaccard_vecinos(vec[a][1], vec[b][1]),
            "error_relativo_sin_alinear": err0, "error_relativo_alineado": err,
            "coseno_PQ": coseno_pq(params[a], params[b])})
    return {k: ep.resumir([p[k] for p in pares if p[k] is not None]) for k in pares[0]}


def resumir_conjunto(conjunto):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    tr = np.array(split.train)
    U, I, R = tr[:, 0].astype(int), tr[:, 1].astype(int), tr[:, 2].astype(float)
    config = f"sgd_lr{ETA}_ep{EPOCAS[conjunto]}_s{ESCALA}"
    filas = [t5.fila_modelo(conjunto, f"{config}_seed{s}", len(ds.idx_to_item), U, I, R)
             for s in SEMILLAS]
    base = ep.POB / conjunto / "part42"
    sesgos = ep.leer(base / "sesgos.json.gz")["recs"]
    j_sesgos = [statistics.fmean(ep.jaccard(x, y) for x, y in zip(
        ep.leer(base / f"{config}_seed{s}.json.gz")["recs"], sesgos)) for s in SEMILLAS]
    return {"epocas": EPOCAS[conjunto], "actualizaciones": EPOCAS[conjunto] * len(R),
            "actualizaciones_linaje_maximas": PASOS, "metricas": t5.media_y_desv(filas),
            "estabilidad": estabilidad(conjunto, config),
            "J_con_sesgos_analiticos": statistics.fmean(j_sesgos)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "resumen":
        ruta = HERE / "control_trayectoria.json"
        total = {c: resumir_conjunto(c) for c in EPOCAS}
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print("guardado")
    else:
        entrenar(sys.argv[1], [int(s) for s in sys.argv[2:]] or SEMILLAS)
