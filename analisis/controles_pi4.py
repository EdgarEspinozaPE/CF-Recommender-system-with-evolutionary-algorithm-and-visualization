"""
Controles de PI-4: ruido de promediar, origen de la coincidencia de listas y
preference match con vectores de ítem permutados.

Uso: python controles_pi4.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py. Salida: controles_pi4.json.
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
import evolutionary_cf as ecf
import explanation_metrics as em

POB = HERE / "poblacion"
GUARDADOS = HERE / "modelos"
SEMILLAS = list(range(42, 47))
TAU_R = ecf.DEFAULT_MIN_EVIDENCE_RATING
SEGMENTOS = [("escasos", 1, 5), ("medios", 6, 20), ("densos", 21, 10**9)]
N_PERMUTACIONES = 3


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def vectores_item(conjunto, id_modelo):
    ruta = POB / conjunto / "part42" / "modelos" / f"{id_modelo}.json.gz"
    d = leer(ruta) if ruta.exists() else json.loads(
        (GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))
    return np.array(d["item_factors"])


def pm_recalculado(Q, usuarios, items, historial):
    """Preference match con la definicion del sistema (hasta 5 items, similitud > 0)."""
    norma = np.linalg.norm(Q, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    X = Q / norma
    pm = np.zeros(len(usuarios))
    n_ev = np.zeros(len(usuarios))
    for k, (u, i) in enumerate(zip(usuarios, items)):
        h = historial.get(u)
        if h is None:
            continue
        sim = X[h[0]] @ X[i]
        val = sim > 0
        if not val.any():
            continue
        s, r = sim[val], h[1][val]
        orden = np.argsort(-s, kind="stable")[:5]
        pm[k] = np.mean(s[orden] * (r[orden] - 1.0) / 4.0)
        n_ev[k] = len(orden)
    return pm, n_ev


def por_segmento(valores, n_int):
    salida = {}
    for nombre, lo, hi in SEGMENTOS:
        m = (n_int >= lo) & (n_int <= hi)
        salida[nombre] = float(valores[m].mean()) if m.any() else None
    return salida


def media_dicts(lista):
    salida = {}
    for k in lista[0]:
        v = [d[k] for d in lista if d[k] is not None]
        salida[k] = statistics.fmean(v) if v else None
    return salida


def control_pm(conjunto, id_modelo, split, interacciones, rng):
    res = leer(POB / conjunto / "part42" / f"{id_modelo}.json.gz")
    f = res["filas"]
    U, I = f["usuario"], f["item"]
    guardado = np.array(f["preference_match"])
    n_int = np.array([interacciones[u] for u in U])
    por_usuario = {}
    for u, i, r in split.train:
        if r >= TAU_R:
            por_usuario.setdefault(u, []).append((i, r))
    historial = {u: (np.array([x[0] for x in v]), np.array([x[1] for x in v], float))
                 for u, v in por_usuario.items()}
    Q = vectores_item(conjunto, id_modelo)
    pm, n_ev = pm_recalculado(Q, U, I, historial)
    error = float(np.max(np.abs(pm - guardado)))
    pm_nulo, ev_nulo, seg_nulo = [], [], []
    for _ in range(N_PERMUTACIONES):
        perm = rng.permutation(len(Q))
        p, e = pm_recalculado(Q[perm], U, I, historial)
        pm_nulo.append(float(p.mean()))
        ev_nulo.append(float(np.mean(e > 0)))
        seg_nulo.append(por_segmento(p, n_int))
    return {"pm": float(guardado.mean()), "error_reproduccion": error,
            "pm_permutado": statistics.fmean(pm_nulo),
            "cobertura_evidencia": float(np.mean(n_ev > 0)),
            "cobertura_evidencia_permutada": statistics.fmean(ev_nulo),
            "pm_por_segmento": por_segmento(guardado, n_int),
            "pm_permutado_por_segmento": media_dicts(seg_nulo)}


def error_tipico(f, clave, poblacion_filas):
    v = np.array(f[clave], float)
    U = np.array(f["usuario"])
    orden = np.argsort(U, kind="stable")
    _, inicio = np.unique(U[orden], return_index=True)
    medias = np.add.reduceat(v[orden], inicio) / np.diff(np.append(inicio, len(v)))
    return float(medias.std(ddof=1) / np.sqrt(len(medias))), float(v.std(ddof=1) / np.sqrt(len(v)))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a or b) else 1.0


def analizar(conjunto):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    ctx = leer(POB / conjunto / "part42" / "contexto.json.gz")
    interacciones = {int(k): v for k, v in ctx["interacciones_train"].items()}
    rng = np.random.default_rng(2026)
    res = {s: leer(POB / conjunto / "part42" / f"evo_dir_s0.3_seed{s}.json.gz") for s in SEMILLAS}
    sesgos = leer(POB / conjunto / "part42" / "sesgos.json.gz")

    ruido = {}
    for clave in ["confidence", "preference_match", "ablation_shift", "necessity", "sufficiency"]:
        medias = [statistics.fmean(res[s]["filas"][clave]) for s in SEMILLAS]
        ts = [error_tipico(res[s]["filas"], clave, None) for s in SEMILLAS]
        ruido[clave] = {"media": statistics.fmean(medias), "desv_semillas": statistics.stdev(medias),
                        "error_tipico_usuario": statistics.fmean(t[0] for t in ts),
                        "error_tipico_recomendacion": statistics.fmean(t[1] for t in ts)}

    j_sesgos = {s: statistics.fmean(jaccard(x, y) for x, y in zip(res[s]["recs"], sesgos["recs"]))
                for s in SEMILLAS}
    fraccion = []
    for a, b in itertools.combinations(SEMILLAS, 2):
        comunes = tambien = 0
        for x, y, z in zip(res[a]["recs"], res[b]["recs"], sesgos["recs"]):
            c = set(x) & set(y)
            comunes += len(c)
            tambien += len(c & set(z))
        fraccion.append(tambien / comunes if comunes else 0.0)

    adoptado = [control_pm(conjunto, f"evo_dir_s0.3_seed{s}", split, interacciones, rng) for s in SEMILLAS]
    print(f"[{conjunto}] control del adoptado listo", flush=True)
    otros = {}
    for id_modelo in ["sgd_lr0.005_ep30_s0.1_seed42", "sgd_lr0.02_ep10_s0.1_seed42", "evo_dir_s0.1_seed42"]:
        otros[id_modelo] = control_pm(conjunto, id_modelo, split, interacciones, rng)
        print(f"[{conjunto}] control de {id_modelo} listo", flush=True)

    def media_control(lista):
        salida = {}
        for k in lista[0]:
            if isinstance(lista[0][k], dict):
                salida[k] = media_dicts([d[k] for d in lista])
            else:
                salida[k] = statistics.fmean(d[k] for d in lista)
        return salida

    return {"ruido_promedio": ruido,
            "jaccard_con_sesgos": statistics.fmean(j_sesgos.values()),
            "comunes_tambien_en_sesgos": statistics.fmean(fraccion),
            "control_pm_adoptado": media_control(adoptado),
            "control_pm_otros": otros}


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"]
    ruta = HERE / "controles_pi4.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        total[c] = analizar(c)
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{c}: guardado", flush=True)
    print(json.dumps(total, indent=1, ensure_ascii=False))
