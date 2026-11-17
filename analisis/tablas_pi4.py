"""
Tablas de PI-4: estabilidad, variabilidad entre semillas y particiones, y
tolerancia a la escasez.

Uso: python tablas_pi4.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py, estabilidad_poblacion.json y
tablas_pi3.json. Salida: tablas_pi4.json.
"""
import gzip
import itertools
import json
import math
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
ADOPTADA = "evo_dir_s0.3"
SGD_ID = "sgd_lr0.005_ep30_s0.1_seed42"
SUBCONJUNTOS = list(itertools.combinations(range(6), 3))
SEGMENTOS = [("escasos", 1, 5), ("medios", 6, 20), ("densos", 21, 10**9)]
PESOS = np.array([0.45, 0.30, 0.25])
UMBRALES = ecf.DEFAULT_CONFIDENCE_THRESHOLDS
TAU_R = ecf.DEFAULT_MIN_EVIDENCE_RATING
K_EVIDENCIA = 3


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, particion, id_modelo):
    ruta = POB / conjunto / f"part{particion}" / "modelos" / f"{id_modelo}.json.gz"
    if ruta.exists():
        return leer(ruta)
    return json.loads((GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))


def categoria(v):
    return np.where(v >= UMBRALES[1], 2, np.where(v >= UMBRALES[0], 1, 0))


def azar_conjuntos(m, a, b):
    """Jaccard esperado entre subconjuntos de tamanos a y b extraidos de m elementos."""
    if a == 0 and b == 0:
        return None
    if a == 0 or b == 0:
        return 0.0
    m = max(m, a, b)
    total = math.comb(m, b)
    esperado = 0.0
    for c in range(max(0, a + b - m), min(a, b) + 1):
        p = math.comb(a, c) * math.comb(m - a, b - c) / total
        esperado += p * c / (a + b - c)
    return esperado


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


def resumen(valores):
    return {"media": statistics.fmean(valores), "desv": statistics.stdev(valores),
            "min": min(valores), "max": max(valores),
            "relativa": statistics.stdev(valores) / abs(statistics.fmean(valores)),
            "valores": valores}


def resumir_dicts(lista):
    return {k: resumen([d[k] for d in lista]) for k in lista[0]}


# ESTABILIDAD Y COMPARACIÓN INDIVIDUAL
def pares_semillas(conjunto, ds, split):
    resultados = {s: leer(POB / conjunto / "part42" / f"{ADOPTADA}_seed{s}.json.gz")
                  for s in SEMILLAS}
    historial = {}
    for u, i, r in split.train:
        historial.setdefault(u, []).append(r)
    n_items = len(ds.idx_to_item)
    tablas = {}
    for s, res in resultados.items():
        f = res["filas"]
        tablas[s] = {(u, i): (conf, pm, nec, tuple(ev)) for u, i, conf, pm, nec, ev in zip(
            f["usuario"], f["item"], f["confidence"], f["preference_match"],
            f["necessity"], f["evidencia_items"])}

    salida = []
    for a, b in itertools.combinations(SEMILLAS, 2):
        comunes = [k for k in tablas[a] if k in tablas[b]]
        d_conf, d_pm, cambia_cat, misma_nec, azar_ev, j_ev = [], [], [], [], [], []
        for k in comunes:
            ca, pa, na, ea = tablas[a][k]
            cb, pb, nb, eb = tablas[b][k]
            d_conf.append(abs(ca - cb))
            d_pm.append(abs(pa - pb))
            cambia_cat.append(categoria(np.array(ca)) != categoria(np.array(cb)))
            misma_nec.append(na == nb)
            m = sum(1 for r in historial.get(k[0], ()) if r >= TAU_R)
            z = azar_conjuntos(m, len(ea), len(eb))
            if z is not None:
                azar_ev.append(z)
                union = set(ea) | set(eb)
                j_ev.append(len(set(ea) & set(eb)) / len(union))
        conf_a = np.array([tablas[a][k][0] for k in comunes])
        conf_b = np.array([tablas[b][k][0] for k in comunes])
        pm_a = np.array([tablas[a][k][1] for k in comunes])
        pm_b = np.array([tablas[b][k][1] for k in comunes])
        solap = statistics.fmean(len(set(x) & set(y)) / cp.TOP_N for x, y in
                                 zip(resultados[a]["recs"], resultados[b]["recs"]))
        salida.append({
            "a": a, "b": b, "comunes": len(comunes), "solapamiento_listas": solap,
            "dif_abs_confianza": statistics.fmean(d_conf),
            "dif_abs_pm": statistics.fmean(d_pm),
            "cambia_categoria": statistics.fmean(map(float, cambia_cat)),
            "misma_necesidad": statistics.fmean(map(float, misma_nec)),
            "corr_confianza": float(np.corrcoef(conf_a, conf_b)[0, 1]),
            "corr_pm": float(np.corrcoef(pm_a, pm_b)[0, 1]),
            "J_evid_verificado": statistics.fmean(j_ev),
            "J_evid_azar": statistics.fmean(azar_ev)})
    ctx = leer(POB / conjunto / "part42" / "contexto.json.gz")
    azar_rec = statistics.fmean(
        azar_conjuntos(n_items - n, cp.TOP_N, cp.TOP_N)
        for n in ctx["interacciones_train"].values())
    return {"por_par": salida,
            "resumen": {k: resumen([p[k] for p in salida]) for k in salida[0] if k not in "ab"},
            "J_rec_azar": azar_rec}


# SEGMENTOS DE ACTIVIDAD
def segmentos_semilla(conjunto, semilla, conocidos, interacciones):
    res = leer(POB / conjunto / "part42" / f"{ADOPTADA}_seed{semilla}.json.gz")
    d = parametros(conjunto, 42, f"{ADOPTADA}_seed{semilla}")
    P, Q = np.array(d["user_factors"]), np.array(d["item_factors"])
    bu, bi, mu = np.array(d["user_bias"]), np.array(d["item_bias"]), d["global_mean"]
    f = res["filas"]
    U, I = np.array(f["usuario"]), np.array(f["item"])
    C = P[U] * Q[I]
    sesgos = mu + bu[U] + bi[I]
    crudo = sesgos + C.sum(axis=1)
    cortes = {}
    for u in dict.fromkeys(U.tolist()):
        s = mu + bu[u] + bi + Q @ P[u]
        s[conocidos[u]] = -np.inf
        cortes[u] = np.partition(s, -11)[-11]
    corte = np.array([cortes[u] for u in U.tolist()])
    orden = np.argsort(-np.abs(C), axis=1)[:, :3]
    dom = C[np.arange(len(C))[:, None], orden].sum(axis=1)
    nec_azar = np.mean([(crudo - C[:, list(S)].sum(axis=1)) < corte for S in SUBCONJUNTOS], axis=0)
    suf_azar = np.mean([(sesgos + C[:, list(S)].sum(axis=1)) >= corte for S in SUBCONJUNTOS], axis=0)
    n_int = np.array([interacciones[u] for u in U.tolist()])
    columnas = {
        "ablation_shift": np.array(f["ablation_shift"]),
        "desplazamiento": np.abs(dom),
        "necesidad": np.array(f["necessity"], float),
        "necesidad_azar": nec_azar,
        "suficiencia": np.array(f["sufficiency"], float),
        "suficiencia_azar": suf_azar,
        "cobertura": np.array(f["coverage"], float),
        "cobertura_intrinseca": np.array(f["intrinsic"], float),
        "cobertura_evidencia": np.array(f["evidencia"], float),
        "evidencia_media": np.array(f["n_evidencia"], float),
        "confianza": np.array(f["confidence"]),
        "preference_match": np.array(f["preference_match"]),
        "senal_usuario": np.array(f["senal_usuario"]),
        "senal_item": np.array(f["senal_item"]),
        "margen": crudo - corte,
        "norma_usuario": np.linalg.norm(P[U], axis=1),
        "abs_bu": np.abs(bu[U]),
    }
    salida = {}
    for nombre, lo, hi in SEGMENTOS:
        m = (n_int >= lo) & (n_int <= hi)
        if not m.any():
            salida[nombre] = None
            continue
        fila = {k: float(v[m].mean()) for k, v in columnas.items()}
        fila["margen_mediana"] = float(np.median(columnas["margen"][m]))
        fila["recortadas"] = float(np.mean(crudo[m] > 5.0))
        fila["ocupa_alta"] = float(np.mean(columnas["confianza"][m] >= UMBRALES[1]))
        fila["ocupa_exploratoria"] = float(np.mean(columnas["confianza"][m] < UMBRALES[0]))
        fila["usuarios"] = int(len(set(U[m].tolist())))
        fila["recomendaciones"] = int(m.sum())
        salida[nombre] = fila
    return salida


def resumir_segmentos(por_semilla):
    salida = {}
    for nombre, _, _ in SEGMENTOS:
        filas = [p[nombre] for p in por_semilla]
        if filas[0] is None:
            salida[nombre] = None
            continue
        salida[nombre] = {k: {"media": statistics.fmean(x[k] for x in filas),
                              "min": min(x[k] for x in filas),
                              "max": max(x[k] for x in filas)} for k in filas[0]}
    return salida


# PARTICIONES
def particiones(conjunto, n_items):
    filas = {"adoptado": [], "sgd": [], "sesgos": []}
    for p in range(42, 47):
        carpeta = POB / conjunto / f"part{p}"
        evo = leer(carpeta / "evo_dir_s0.3_seed42.json.gz")
        sgd = leer(carpeta / f"{SGD_ID}.json.gz")
        ses = leer(carpeta / "sesgos.json.gz")
        filas["adoptado"].append({**calidad(evo, n_items), **explicativas(evo["filas"])})
        filas["sgd"].append({**calidad(sgd, n_items), **explicativas(sgd["filas"])})
        filas["sesgos"].append(calidad(ses, n_items))
    salida = {k: resumir_dicts(v) for k, v in filas.items()}
    salida["diferencias"] = {
        "rmse_adoptado_menos_sesgos": [a["rmse"] - b["rmse"] for a, b in zip(filas["adoptado"], filas["sesgos"])],
        "ndcg_adoptado_menos_sesgos": [a["ndcg10"] - b["ndcg10"] for a, b in zip(filas["adoptado"], filas["sesgos"])],
        "rmse_adoptado_menos_sgd": [a["rmse"] - b["rmse"] for a, b in zip(filas["adoptado"], filas["sgd"])],
        "ndcg_adoptado_menos_sgd": [a["ndcg10"] - b["ndcg10"] for a, b in zip(filas["adoptado"], filas["sgd"])],
    }
    return salida


def analizar(conjunto):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    n_items = len(ds.idx_to_item)
    conocidos = {}
    for u, i, _ in split.train:
        conocidos.setdefault(u, []).append(i)
    ctx = leer(POB / conjunto / "part42" / "contexto.json.gz")
    interacciones = {int(k): v for k, v in ctx["interacciones_train"].items()}
    n_vals = np.array(list(interacciones.values()))

    semillas = []
    for s in SEMILLAS:
        res = leer(POB / conjunto / "part42" / f"{ADOPTADA}_seed{s}.json.gz")
        semillas.append({**calidad(res, n_items), **explicativas(res["filas"])})
    pi3 = json.loads((HERE / "tablas_pi3.json").read_text(encoding="utf-8"))[conjunto]["semillas"]
    for fila, otra in zip(semillas, pi3):
        fila["corr_pm_score"] = otra["corr_pm_score"]
        fila["corr_pm_latente"] = otra["corr_pm_latente"]

    por_semilla = [segmentos_semilla(conjunto, s, conocidos, interacciones) for s in SEMILLAS]
    print(f"[{conjunto}] segmentos listos", flush=True)
    salida = {
        "actividad": {"minimo": int(n_vals.min()), "mediana": float(np.median(n_vals)),
                      "maximo": int(n_vals.max()), "usuarios": len(n_vals),
                      "por_segmento": {n: int(((n_vals >= lo) & (n_vals <= hi)).sum())
                                       for n, lo, hi in SEGMENTOS}},
        "estabilidad": json.loads((HERE / "estabilidad_poblacion.json").read_text(encoding="utf-8"))
        [conjunto]["configuraciones"][ADOPTADA]["resumen"],
        "pares": pares_semillas(conjunto, ds, split),
        "variabilidad_semillas": resumir_dicts(semillas),
        "segmentos": resumir_segmentos(por_semilla),
    }
    if conjunto == "movielens":
        salida["particiones"] = particiones(conjunto, n_items)
    return salida


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"]
    ruta = HERE / "tablas_pi4.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        total[c] = analizar(c)
        ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{c}: guardado", flush=True)
