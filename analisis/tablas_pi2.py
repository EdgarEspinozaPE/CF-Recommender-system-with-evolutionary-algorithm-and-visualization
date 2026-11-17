"""
Cifras de PI-2 y de la sección de usuarios evaluados: selección del caso típico
y proporciones sobre toda la población.

Uso: python tablas_pi2.py [conjunto ...]    (por defecto, los tres)
Entrada: resultados de corridas_poblacion.py. Salida: tablas_pi2.json.
"""
import gzip
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
import explanation_metrics as em
from evolutionary_cf import DEFAULT_MIN_EVIDENCE_RATING

POB = HERE / "poblacion"
GUARDADOS = HERE / "modelos"
ADOPTADO = "evo_dir_s0.3"
SEMILLAS = range(42, 47)
CLAVES = ["confidence", "preference_match", "ablation_shift"]
UMBRAL_INACTIVA = 0.01


def leer_gz(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def parametros(conjunto, semilla):
    ruta = POB / conjunto / "part42" / "modelos" / f"{ADOPTADO}_seed{semilla}.json.gz"
    if ruta.exists():
        return leer_gz(ruta)
    return json.loads((GUARDADOS / f"{conjunto}_seed42_train.json").read_text(encoding="utf-8"))


def filas(conjunto, semilla):
    return leer_gz(POB / conjunto / "part42" / f"{ADOPTADO}_seed{semilla}.json.gz")["filas"]


def por_usuario(f, clave):
    out = {}
    for u, v in zip(f["usuario"], f[clave]):
        out.setdefault(u, []).append(v)
    return out


def indexado(valor):
    if isinstance(valor, dict):
        return {int(k): v for k, v in valor.items()}
    return dict(enumerate(valor))


# USUARIOS EVALUADOS Y EFECTO DE LA CONDICIÓN DE PRUEBA
def usuarios_evaluados(conjunto, ctx, f):
    con_relevante = set(ctx["con_relevante_en_prueba"])
    efecto = {}
    for m in CLAVES + ["necessity"]:
        medias = {u: statistics.fmean(v) for u, v in por_usuario(f, m).items()}
        todos = statistics.fmean(medias.values())
        con = statistics.fmean(v for u, v in medias.items() if u in con_relevante)
        efecto[m] = {"poblacion": todos, "con_condicion": con,
                     "diferencia_relativa": (con - todos) / todos}
    return {"poblacion": len(ctx["poblacion"]), "recomendaciones": len(f["usuario"]),
            "usuarios_ranking": len(ctx["usuarios_ranking"]),
            "excluidos_por_la_condicion": 1 - len(ctx["usuarios_ranking"]) / len(ctx["poblacion"]),
            "n_train": ctx["n_train"], "n_test": ctx["n_test"],
            "efecto_de_la_condicion": efecto}


# CASO TÍPICO
def caso_tipico(ctx, f):
    perfil = {}
    for m in CLAVES:
        for u, v in por_usuario(f, m).items():
            perfil.setdefault(u, {})[m] = statistics.fmean(v)
    medianas = {m: statistics.median(p[m] for p in perfil.values()) for m in CLAVES}
    desv = {m: statistics.pstdev([p[m] for p in perfil.values()]) or 1.0 for m in CLAVES}
    dist = {u: sum(abs(p[m] - medianas[m]) / desv[m] for m in CLAVES) for u, p in perfil.items()}
    orden = sorted(perfil, key=dist.get)
    n_train = indexado(ctx["interacciones_train"])
    ids = indexado(ctx["ids_usuario"])
    historiales = sorted(n_train[u] for u in perfil)

    def percentil(u):
        return 100 * sum(x <= n_train[u] for x in historiales) / len(historiales)

    return {"medianas": medianas,
            "mediana_interacciones_train": statistics.median(historiales),
            "primeros": [{"usuario": ids[u], "puesto": k, "distancia": dist[u],
                          "n_train": n_train[u], "percentil_historial": percentil(u),
                          **perfil[u]} for k, u in enumerate(orden[:5], 1)]}


# PROPORCIONES SOBRE LA POBLACIÓN
def recorte(f):
    crudo = por_usuario(f, "raw_score")
    n = len(f["raw_score"])
    return {"recomendaciones_recortadas": sum(r >= 5.0 for r in f["raw_score"]) / n,
            "usuarios_con_las_diez_recortadas":
                sum(all(r >= 5.0 for r in v) for v in crudo.values()) / len(crudo)}


def afinidad_frente_a_sesgo(conjunto):
    out = []
    for s in SEMILLAS:
        bi = parametros(conjunto, s)["item_bias"]
        f = filas(conjunto, s)
        out.append(statistics.fmean(abs(l) > abs(bi[i])
                                    for l, i in zip(f["latent_dot"], f["item"])))
    return statistics.fmean(out)


def dimensiones(conjunto):
    inactivas, rangos = [], []
    for s in SEMILLAS:
        d = parametros(conjunto, s)
        pu, qi = np.array(d["user_factors"]), np.array(d["item_factors"])
        items = por_usuario(filas(conjunto, s), "item")
        n = sum(int(np.any(np.max(np.abs(pu[u][None, :] * qi[it[:5]]), axis=0) < UMBRAL_INACTIVA))
                for u, it in items.items())
        inactivas.append(n / len(items))
        _, rp = np.linalg.qr(pu)
        _, rq = np.linalg.qr(qi)
        energia = np.linalg.svd(rp @ rq.T, compute_uv=False) ** 2
        energia = energia / energia.sum()
        rangos.append(math.exp(-float(np.sum(energia * np.log(energia)))))
    return {"usuarios_con_alguna_dimension_inactiva": statistics.fmean(inactivas),
            "rango_efectivo": statistics.fmean(rangos)}


def evidencia(conjunto, f):
    ds = cp.CONJUNTOS[conjunto][0]()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    valores = [r for _, _, r in split.train]
    n = f["n_evidencia"]
    return {"media_por_recomendacion": statistics.fmean(n),
            "recomendaciones_con_dos_o_menos": sum(x <= 2 for x in n) / len(n),
            "interacciones_sobre_el_umbral":
                sum(r >= DEFAULT_MIN_EVIDENCE_RATING for r in valores) / len(valores)}


def main(conjuntos):
    ruta = HERE / "tablas_pi2.json"
    total = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    for c in conjuntos:
        ctx = leer_gz(POB / c / "part42" / "contexto.json.gz")
        f = filas(c, 42)
        total[c] = {"usuarios_evaluados": usuarios_evaluados(c, ctx, f),
                    "caso_tipico": caso_tipico(ctx, f),
                    "recorte": recorte(f),
                    "afinidad_mayor_que_sesgo_de_item": afinidad_frente_a_sesgo(c),
                    "dimensiones": dimensiones(c),
                    "evidencia": evidencia(c, f)}
        t = total[c]
        primero = t["caso_tipico"]["primeros"][0]
        print(f"[{c}] población {t['usuarios_evaluados']['poblacion']}, "
              f"ranking {t['usuarios_evaluados']['usuarios_ranking']}; "
              f"caso típico {primero['usuario']} (distancia {primero['distancia']:.3f}, "
              f"n_train {primero['n_train']}); recortadas {t['recorte']['recomendaciones_recortadas']:.3f}; "
              f"afinidad > sesgo {t['afinidad_mayor_que_sesgo_de_item']:.3f}; "
              f"dimensión inactiva {t['dimensiones']['usuarios_con_alguna_dimension_inactiva']:.3f}; "
              f"rango efectivo {t['dimensiones']['rango_efectivo']:.2f}")
    ruta.write_text(json.dumps(total, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["movielens", "bookcrossing", "lastfm"])
