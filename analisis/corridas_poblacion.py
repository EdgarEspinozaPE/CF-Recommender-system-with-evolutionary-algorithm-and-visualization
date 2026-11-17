"""
Entrena y evalúa los modelos de la tesis sobre toda la población de usuarios.

Uso: python corridas_poblacion.py <conjunto> <grupo> [<grupo> ...]
  conjunto     movielens | lastfm | bookcrossing
  grupo        base         partición 42: modelo adoptado (semillas 42-46),
                            alternativas de la ablación, SGD y solo sesgos
               gamma        partición 42: aptitud explicativa, semillas 42-46
                            (solo movielens)
               particiones  particiones 43-46: modelo adoptado, SGD y solo
                            sesgos (solo movielens)

Salida: poblacion/<conjunto>/part<p>/<id>.json.gz. Lo ya calculado no se repite.
"""
import gzip
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import evolutionary_cf as ecf
import explanation_metrics as em
from baseline_sgd import SGDMatrixFactorization
from evolutionary_cf import (
    EvolutionaryCollaborativeFilter,
    Individual,
    build_recommendation_evidence,
    known_items_by_user,
    load_bookcrossing_folder,
    load_lastfm_hetrec_folder,
    load_movielens_100k,
)

DATA = HERE.parent / "datasets"
RAIZ = Path(os.environ.get("POB_SALIDA", HERE / "poblacion"))
GENERACIONES = int(os.environ.get("POB_GENERACIONES", "120"))
LIMITE = int(os.environ.get("POB_LIMITE", "0"))
GUARDADOS = HERE / "modelos"
TOP_N = 10

CONJUNTOS = {
    "movielens": (lambda: load_movielens_100k(DATA / "ml-100k"), 4.0),
    "lastfm": (lambda: load_lastfm_hetrec_folder(
        DATA / "hetrec-lastfm", min_user_interactions=5, min_item_interactions=5), 3.0),
    "bookcrossing": (lambda: load_bookcrossing_folder(
        DATA, min_user_interactions=5, min_item_interactions=5), 4.0),
}
SGD_CONFIGS = [(0.02, 10, 0.1), (0.005, 30, 0.1), (0.005, 10, 0.1)]
SGD_MENOR_RMSE = (0.005, 30, 0.1)


# CACHÉ DE CONSULTAS REPETIDAS
_cache_usuario, _cache_items = {}, {}
_original_ratings_for_user = ecf.ratings_for_user
_original_item_counts = ecf.item_rating_counts


def _ratings_for_user(ratings, user_idx):
    clave = id(ratings)
    if clave not in _cache_usuario:
        por = {}
        for u, i, r in ratings:
            por.setdefault(u, []).append((i, r))
        _cache_usuario[clave] = (ratings, por)
    return list(_cache_usuario[clave][1].get(user_idx, []))


def _item_rating_counts(ratings):
    clave = id(ratings)
    if clave not in _cache_items:
        _cache_items[clave] = (ratings, _original_item_counts(ratings))
    return dict(_cache_items[clave][1])


ecf.ratings_for_user = _ratings_for_user
ecf.item_rating_counts = _item_rating_counts


# ESPECIFICACIONES DE LOS MODELOS
def evo(guia, escala, semilla, gamma=0.0):
    tipo = "dir" if guia > 0 else "nodir"
    sufijo = f"_gam{gamma}" if gamma else ""
    return {"id": f"evo_{tipo}_s{escala}{sufijo}_seed{semilla}", "tipo": "evo",
            "guia": guia, "escala": escala, "gamma": gamma, "semilla": semilla}


def sgd(lr, epocas, escala, semilla):
    return {"id": f"sgd_lr{lr}_ep{epocas}_s{escala}_seed{semilla}", "tipo": "sgd",
            "lr": lr, "epocas": epocas, "escala": escala, "semilla": semilla}


SESGOS = {"id": "sesgos", "tipo": "sesgos"}


def plan(grupo):
    """Lista de (particion, especificacion) de un grupo."""
    if grupo == "base":
        specs = [evo(0.01, 0.3, s) for s in range(42, 47)]
        specs += [evo(0.0, 0.4082, s) for s in (42, 43)]
        specs += [evo(0.01, e, s) for e in (0.1, 0.4082) for s in (42, 43)]
        specs += [sgd(lr, ep, e, s) for lr, ep, e in SGD_CONFIGS for s in (42, 43)]
        specs.append(SESGOS)
        return [(42, s) for s in specs]
    if grupo == "gamma":
        return [(42, evo(0.01, 0.3, s, g)) for g in (0.5, 2.0) for s in range(42, 47)]
    if grupo == "particiones":
        out = []
        for p in range(43, 47):
            out += [(p, evo(0.01, 0.3, 42)), (p, sgd(*SGD_MENOR_RMSE, 42)), (p, SESGOS)]
        return out
    raise SystemExit(f"grupo desconocido: {grupo}")


# CONTEXTO DE CADA PARTICIÓN
_contextos = {}


def contexto(nombre, particion):
    clave = (nombre, particion)
    if clave in _contextos:
        return _contextos[clave]
    cargar, tau = CONJUNTOS[nombre]
    ds = cargar()
    split = em.split_train_test_by_user(ds.ratings, 0.20, particion)
    train_ds = em.dataset_with_ratings(ds, split.train)
    known_map = known_items_by_user(train_ds.ratings)
    poblacion = em.sample_user_indices(ds, 10**6, TOP_N, 42, train_ratings=split.train)
    ranking = em.sample_user_indices(ds, 10**6, TOP_N, 42, train_ratings=split.train,
                                     test_ratings=split.test, relevant_threshold=tau)
    if LIMITE:
        poblacion = poblacion[:LIMITE]
        ranking = [u for u in ranking if u in set(poblacion)]
    relevantes = em.relevant_items_by_user(split.test, tau)
    ctx = {"ds": ds, "split": split, "train_ds": train_ds, "known_map": known_map,
           "poblacion": poblacion, "ranking": ranking, "relevantes": relevantes,
           "tau": tau}
    carpeta = RAIZ / nombre / f"part{particion}"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "contexto.json.gz"
    if not ruta.exists():
        escribir(ruta, {
            "conjunto": nombre, "particion": particion, "tau_rel": tau,
            "ids_usuario": ds.idx_to_user, "ids_item": ds.idx_to_item,
            "n_train": len(split.train), "n_test": len(split.test),
            "interacciones_train": {u: len(known_map.get(u, ())) for u in poblacion},
            "poblacion": poblacion, "usuarios_ranking": ranking,
            "con_relevante_en_prueba": [u for u in poblacion if relevantes.get(u)],
            "test": [[u, i, r] for u, i, r in split.test]})
    _contextos[clave] = ctx
    return ctx


def escribir(ruta, datos):
    tmp = ruta.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(ruta)


def r6(x):
    return None if x is None else round(float(x), 6)


# CONSTRUCCIÓN DE MODELOS
def construir_evo(ctx, spec, generaciones=None):
    ds = ctx["ds"]
    kw = dict(n_users=len(ds.idx_to_user), n_items=len(ds.idx_to_item), factors=6,
              population_size=25, generations=generaciones or GENERACIONES,
              mutation_sigma=0.08, mutation_rate=0.02, crossover_rate=0.60,
              guidance_rate=spec["guia"], guidance_batch=8192,
              bias_guidance_ratio=1.0, init_scale=spec["escala"],
              seed=spec["semilla"], verbose=False)
    if spec["gamma"]:
        kw.update(explainability_weight=spec["gamma"], explainability_sample=8,
                  explainability_top_n=TOP_N)
    return EvolutionaryCollaborativeFilter(**kw)


def modelo_para(nombre, particion, ctx, spec):
    """Devuelve (modelo, segundos de entrenamiento, reutilizado)."""
    ds, train = ctx["ds"], ctx["split"].train
    if spec["tipo"] == "evo":
        guardado = GUARDADOS / f"{nombre}_seed42_train.json"
        if (particion == 42 and spec["id"] == "evo_dir_s0.3_seed42"
                and guardado.exists() and GENERACIONES == 120 and not LIMITE):
            m = construir_evo(ctx, spec)
            m.best_ = Individual(**json.loads(guardado.read_text(encoding="utf-8")))
            return m, None, True
        t = time.time()
        m = construir_evo(ctx, spec).fit(train)
        return m, time.time() - t, False
    if spec["tipo"] == "sgd":
        t = time.time()
        m = SGDMatrixFactorization(
            n_users=len(ds.idx_to_user), n_items=len(ds.idx_to_item),
            learning_rate=spec["lr"], epochs=spec["epocas"],
            init_scale=spec["escala"], seed=spec["semilla"]).fit(train)
        return m, time.time() - t, False
    m = construir_evo(ctx, evo(0.0, 0.3, 42), generaciones=1)
    mu, bu, bi = m._baseline_terms(train)
    m.best_ = Individual(
        user_factors=[[0.0] * 6 for _ in range(m.n_users)],
        item_factors=[[0.0] * 6 for _ in range(m.n_items)],
        user_bias=list(bu), item_bias=list(bi), global_mean=mu)
    return m, 0.0, False


# EVALUACIÓN
def evaluar(modelo, ctx, explicar):
    split, known_map, train_ds = ctx["split"], ctx["known_map"], ctx["train_ds"]
    errores = [r6(r - modelo.predict(u, i)) for u, i, r in split.test]

    b = modelo.best_
    paso = max(1, modelo.n_items // 240)
    columnas = {k: [] for k in [
        "usuario", "item", "raw_score", "latent_dot", "ablation_shift", "necessity",
        "sufficiency", "intrinsic", "evidencia", "coverage", "confidence",
        "preference_match", "n_evidencia", "senal_usuario", "senal_item",
        "factores", "evidencia_items"]}
    recs_pob, lat_sd = [], []
    for u in ctx["poblacion"]:
        recs, corte = modelo.recommend_with_cutoff(u, known_map.get(u, set()), TOP_N)
        recs_pob.append([i for i, _ in recs])
        pu = b.user_factors[u]
        lat = [sum(x * y for x, y in zip(pu, b.item_factors[i]))
               for i in range(0, modelo.n_items, paso)]
        lat_sd.append(r6(statistics.pstdev(lat)))
        if not explicar:
            continue
        evid = build_recommendation_evidence(modelo, train_ds, u, recs)
        filas = em.evaluate_user_recommendations(modelo, u, recs, evid,
                                                 next_best_raw_score=corte)
        for f, e in zip(filas, evid):
            columnas["usuario"].append(u)
            columnas["item"].append(int(f["item_idx"]))
            columnas["raw_score"].append(r6(f["raw_score"]))
            columnas["latent_dot"].append(r6(e["components"]["latent_dot"]))
            columnas["ablation_shift"].append(r6(f["ablation_shift"]))
            columnas["necessity"].append(int(f["necessity"]))
            columnas["sufficiency"].append(int(f["sufficiency"]))
            columnas["intrinsic"].append(int(f["intrinsic_explanation"]))
            columnas["evidencia"].append(int(f["user_evidence_explanation"]))
            columnas["coverage"].append(int(f["coverage"]))
            columnas["confidence"].append(r6(f["confidence"]))
            columnas["preference_match"].append(r6(f["preference_match"]))
            columnas["n_evidencia"].append(int(f["similar_evidence_count"]))
            columnas["senal_usuario"].append(r6(e["confidence_signals"][1]))
            columnas["senal_item"].append(r6(e["confidence_signals"][2]))
            columnas["factores"].append(list(f["top_factor_signature"]))
            columnas["evidencia_items"].append(list(f["similar_item_signature"]))

    posicion = {u: k for k, u in enumerate(ctx["poblacion"])}
    ranking = {k: [] for k in ["usuario", "p10", "r10", "ndcg10", "acierto", "diversidad"]}
    for u in ctx["ranking"]:
        recs = recs_pob[posicion[u]]
        rel = ctx["relevantes"].get(u, set())
        aciertos = len(set(recs) & rel)
        ranking["usuario"].append(u)
        ranking["p10"].append(r6(aciertos / TOP_N))
        ranking["r10"].append(r6(aciertos / len(rel)))
        ranking["ndcg10"].append(r6(em.ndcg_at_k(recs, rel, TOP_N)))
        ranking["acierto"].append(1 if aciertos else 0)
        ranking["diversidad"].append(r6(em.intra_list_diversity(modelo, recs))
                                     if explicar else None)
    return {"errores": errores, "recs": recs_pob, "lat_sd": lat_sd,
            "ranking": ranking, "filas": columnas if explicar else None}


def resumen(res):
    e = res["errores"]
    out = {"rmse": statistics.fmean(x * x for x in e) ** 0.5,
           "mae": statistics.fmean(abs(x) for x in e),
           "p10": statistics.fmean(res["ranking"]["p10"]),
           "ndcg10": statistics.fmean(res["ranking"]["ndcg10"])}
    f = res["filas"]
    if f:
        for k in ["confidence", "preference_match", "ablation_shift", "necessity"]:
            out[k] = statistics.fmean(f[k])
    return {k: r6(v) for k, v in out.items()}


def correr(nombre, grupos):
    for grupo in grupos:
        for particion, spec in plan(grupo):
            if nombre != "movielens" and grupo in ("gamma", "particiones"):
                continue
            carpeta = RAIZ / nombre / f"part{particion}"
            ruta = carpeta / f"{spec['id']}.json.gz"
            if ruta.exists():
                print(f"[{nombre} p{particion}] {spec['id']}: ya existe", flush=True)
                continue
            ctx = contexto(nombre, particion)
            t0 = time.time()
            modelo, seg, reutilizado = modelo_para(nombre, particion, ctx, spec)
            if not reutilizado and spec["tipo"] != "sesgos":
                b = modelo.best_
                (carpeta / "modelos").mkdir(exist_ok=True)
                escribir(carpeta / "modelos" / f"{spec['id']}.json.gz", {
                    "global_mean": b.global_mean, "user_bias": b.user_bias,
                    "item_bias": b.item_bias, "user_factors": b.user_factors,
                    "item_factors": b.item_factors, "fitness": b.fitness})
            t1 = time.time()
            res = evaluar(modelo, ctx, explicar=spec["tipo"] != "sesgos")
            res.update({"conjunto": nombre, "particion": particion, "spec": spec,
                        "segundos_entrenamiento": seg, "reutilizado": reutilizado,
                        "segundos_evaluacion": time.time() - t1})
            res["resumen"] = resumen(res)
            escribir(ruta, res)
            print(f"[{nombre} p{particion}] {spec['id']}: "
                  f"entreno {'reutilizado' if reutilizado else f'{seg:.0f} s'}, "
                  f"evaluacion {time.time() - t1:.0f} s, total {time.time() - t0:.0f} s "
                  f"{res['resumen']}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in CONJUNTOS:
        raise SystemExit(__doc__)
    correr(sys.argv[1], sys.argv[2:])
