"""
Tablas de PI-3: fidelidad de las explicaciones.

Uso: python tablas_pi3.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py. Salida: tablas_pi3.json.
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
SEMILLAS = range(42, 47)
SUBCONJUNTOS = list(itertools.combinations(range(6), 3))
PESOS = [((0.45, 0.30, 0.25), "adoptada"), ((1 / 3, 1 / 3, 1 / 3), "uniforme"),
         ((0.60, 0.20, 0.20), "afinidad"), ((0.20, 0.40, 0.40), "volumen"),
         ((1.0, 0.0, 0.0), "solo_afinidad"), ((0.0, 0.5, 0.5), "sin_afinidad")]
UMBRALES = [(0.25, 0.50), (0.30, 0.60), (1 / 3, 2 / 3), (0.40, 0.70), (0.50, 0.75)]


def leer(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, semilla):
    ruta = POB / conjunto / "part42" / "modelos" / f"evo_dir_s0.3_seed{semilla}.json.gz"
    if ruta.exists():
        return leer(ruta)
    return json.loads((GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))


def calidad(res, n_items):
    e = np.array(res["errores"])
    r = res["ranking"]
    salida = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
              "p10": statistics.fmean(r["p10"]), "r10": statistics.fmean(r["r10"]),
              "ndcg10": statistics.fmean(r["ndcg10"]), "hit10": statistics.fmean(r["acierto"]),
              "cobertura_catalogo": len({i for lista in res["recs"] for i in lista}) / n_items}
    dv = [x for x in r["diversidad"] if x is not None]
    if dv:
        salida["diversidad"] = statistics.fmean(dv)
    return salida


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else None


def intra_usuario(grupos, a, b):
    valores = [pearson(a[g], b[g]) for g in grupos if len(g) >= 3]
    valores = [v for v in valores if v is not None]
    return statistics.fmean(valores)


def rangos(v):
    """Rangos medios, para que los empates no dependan del orden de los datos."""
    _, inv, cuenta = np.unique(v, return_inverse=True, return_counts=True)
    acumulado = np.cumsum(cuenta)
    medio = acumulado - (cuenta - 1) / 2.0
    return medio[inv]


def categoria(v, umbrales):
    return np.where(v >= umbrales[1], 2, np.where(v >= umbrales[0], 1, 0))


def analizar_semilla(conjunto, semilla, conocidos, n_items):
    res = leer(POB / conjunto / "part42" / f"evo_dir_s0.3_seed{semilla}.json.gz")
    d = parametros(conjunto, semilla)
    P, Q = np.array(d["user_factors"]), np.array(d["item_factors"])
    bu, bi, mu = np.array(d["user_bias"]), np.array(d["item_bias"]), d["global_mean"]
    f = res["filas"]
    U, I = np.array(f["usuario"]), np.array(f["item"])
    guardado = np.array(f["raw_score"])

    C = P[U] * Q[I]
    sesgos = mu + bu[U] + bi[I]
    crudo = sesgos + C.sum(axis=1)
    assert np.max(np.abs(crudo - guardado)) < 2e-6, "score crudo distinto del guardado"

    cortes = {}
    usuarios = list(dict.fromkeys(U.tolist()))
    for u in usuarios:
        s = mu + bu[u] + bi + Q @ P[u]
        s[conocidos[u]] = -np.inf
        cortes[u] = np.partition(s, -11)[-11]
    corte = np.array([cortes[u] for u in U.tolist()])
    margen = crudo - corte

    orden = np.argsort(-np.abs(C), axis=1)[:, :3]
    filas_idx = np.arange(len(C))[:, None]
    dom = C[filas_idx, orden].sum(axis=1)
    nec = (crudo - dom) < corte
    suf = (sesgos + dom) >= corte
    nec_g, suf_g = np.array(f["necessity"], bool), np.array(f["sufficiency"], bool)
    coincide_nec = float(np.mean(nec == nec_g))
    coincide_suf = float(np.mean(suf == suf_g))

    nec_azar, suf_azar, ambas_azar = [], [], []
    for S in SUBCONJUNTOS:
        parte = C[:, S].sum(axis=1)
        nec_s = (crudo - parte) < corte
        suf_s = (sesgos + parte) >= corte
        nec_azar.append(float(np.mean(nec_s)))
        suf_azar.append(float(np.mean(suf_s)))
        ambas_azar.append(float(np.mean(nec_s & suf_s)))

    grupos = []
    inicio = 0
    for k in range(1, len(U) + 1):
        if k == len(U) or U[k] != U[inicio]:
            grupos.append(np.arange(inicio, k))
            inicio = k

    pm = np.array(f["preference_match"])
    lat = C.sum(axis=1)
    ib = bi[I]
    sd_lat = statistics.fmean(float(lat[g].std()) for g in grupos if len(g) >= 3)
    sd_bi = statistics.fmean(float(ib[g].std()) for g in grupos if len(g) >= 3)

    conf = np.array(f["confidence"])
    senal_u, senal_i = np.array(f["senal_usuario"]), np.array(f["senal_item"])
    senales_m = np.vstack([pm, senal_u, senal_i])
    base = 0.45 * pm + 0.30 * senal_u + 0.25 * senal_i
    umbral_base = UMBRALES[2]
    rango_base = rangos(base)
    pesos = {}
    for w, nombre in PESOS:
        v = np.array(w) @ senales_m
        pesos[nombre] = {"media": float(v.mean()), "desv": float(v.std()),
                         "spearman": pearson(rango_base, rangos(v)),
                         "cambian": float(np.mean(categoria(v, umbral_base)
                                                  != categoria(base, umbral_base)))}
    umbrales = {}
    for u_ in UMBRALES:
        c_ = categoria(base, u_)
        umbrales[f"{u_[0]:.4f}/{u_[1]:.4f}"] = {"exploratoria": float(np.mean(c_ == 0)),
                                                "media": float(np.mean(c_ == 1)),
                                                "alta": float(np.mean(c_ == 2))}

    modelo = ecf.EvolutionaryCollaborativeFilter(n_users=len(P), n_items=len(Q), factors=6, seed=semilla)
    modelo.best_ = ecf.Individual(**d)
    error_rec = 0.0
    for u, i in zip(U.tolist(), I.tolist()):
        c = modelo.prediction_components(u, i)
        error_rec = max(error_rec, abs(c["global_mean"] + c["user_bias"] + c["item_bias"]
                                       + sum(c["factor_contributions"]) - c["raw_prediction"]))

    amplitud = statistics.fmean(float(crudo[g].max() - crudo[g].min()) for g in grupos)
    salida = {
        "calidad": calidad(res, n_items),
        "recomendaciones": len(U), "usuarios": len(usuarios),
        "error_reconstruccion": error_rec,
        "ablation_shift": statistics.fmean(f["ablation_shift"]),
        "desplazamiento_puntos": float(np.mean(np.abs(dom))),
        "necesidad": float(nec_g.mean()), "suficiencia": float(suf_g.mean()),
        "coincide_necesidad": coincide_nec, "coincide_suficiencia": coincide_suf,
        "necesaria_y_suficiente": float(np.mean(nec_g & suf_g)),
        "solo_suficiente": float(np.mean(~nec_g & suf_g)),
        "solo_necesaria": float(np.mean(nec_g & ~suf_g)),
        "ninguna": float(np.mean(~nec_g & ~suf_g)),
        "necesidad_azar": statistics.fmean(nec_azar),
        "suficiencia_azar": statistics.fmean(suf_azar),
        "ambas_azar": statistics.fmean(ambas_azar),
        "necesidad_azar_min": min(nec_azar), "necesidad_azar_max": max(nec_azar),
        "sostenida_solo_por_sesgos": float(np.mean(sesgos >= corte)),
        "no_necesaria_sostenida_por_sesgos": float(np.mean((sesgos >= corte)[~nec_g])) if (~nec_g).any() else None,
        "amplitud_top10": amplitud,
        "margen_medio": float(margen.mean()), "margen_mediana": float(np.median(margen)),
        "cobertura": statistics.fmean(f["coverage"]),
        "cobertura_intrinseca": statistics.fmean(f["intrinsic"]),
        "cobertura_evidencia": statistics.fmean(f["evidencia"]),
        "evidencia_media": statistics.fmean(f["n_evidencia"]),
        "confianza_media": float(conf.mean()), "confianza_desv": float(conf.std()),
        "confianza_min": float(conf.min()), "confianza_max": float(conf.max()),
        "preference_match": float(pm.mean()), "preference_match_desv": float(pm.std()),
        "corr_score_pm_global": pearson(crudo, pm),
        "corr_pm_latente": intra_usuario(grupos, pm, lat),
        "corr_pm_sesgo_item": intra_usuario(grupos, pm, ib),
        "corr_pm_score": intra_usuario(grupos, pm, crudo),
        "corr_score_latente": intra_usuario(grupos, crudo, lat),
        "corr_score_sesgo_item": intra_usuario(grupos, crudo, ib),
        "aporte_latente": sd_lat / (sd_lat + sd_bi),
        "pesos": pesos, "umbrales": umbrales,
        "senales": {
            "afinidad": {"media": float(pm.mean()), "desv": float(pm.std()), "corr": pearson(pm, base)},
            "usuario": {"media": float(senal_u.mean()), "desv": float(senal_u.std()), "corr": pearson(senal_u, base)},
            "item": {"media": float(senal_i.mean()), "desv": float(senal_i.std()), "corr": pearson(senal_i, base)},
        },
    }
    return salida


def resumir(por_semilla):
    """Media y recorrido entre semillas de cada numero, conservando la estructura."""
    muestra = por_semilla[0]
    if isinstance(muestra, dict):
        return {k: resumir([p[k] for p in por_semilla]) for k in muestra}
    if muestra is None:
        return None
    return {"media": statistics.fmean(por_semilla), "min": min(por_semilla), "max": max(por_semilla)}


def analizar(conjunto):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    n_items = len(ds.idx_to_item)
    conocidos = {}
    for u, i, _ in split.train:
        conocidos.setdefault(u, []).append(i)
    por_semilla = []
    for s in SEMILLAS:
        por_semilla.append(analizar_semilla(conjunto, s, conocidos, n_items))
        print(f"[{conjunto}] semilla {s} lista", flush=True)
    sesgos = leer(POB / conjunto / "part42" / "sesgos.json.gz")
    return {"n_prueba": len(split.test), "n_entrenamiento": len(split.train),
            "sesgos": calidad(sesgos, n_items),
            "semillas": por_semilla, "resumen": resumir(por_semilla)}


if __name__ == "__main__":
    conjuntos = sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"]
    ruta = HERE / "tablas_pi3.json"
    salida = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        salida[c] = analizar(c)
        ruta.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    print("escrito", ruta)
