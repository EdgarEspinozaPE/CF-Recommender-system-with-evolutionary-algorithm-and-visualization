"""
Figuras y datos de los casos de estudio, un usuario por conjunto.

Uso: python casos_estudio.py <conjunto>
  conjunto   movielens | lastfm | bookcrossing

Salida: casos_estudio/<conjunto>/ (figuras y datos.json).
Con CASOS_REENTRENAR=1 se repite el entrenamiento en lugar de reutilizar datos.json.
"""
import gzip
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import corridas_poblacion as cp
import evolutionary_cf as ecf
import explanation_metrics as em
from evolutionary_cf import (
    EvolutionaryCollaborativeFilter,
    Individual,
    build_recommendation_evidence,
    cosine_similarity,
    item_label,
    known_items_by_user,
    short_label,
    two_factor_point,
)

TOP_N, K = 10, 6
GENERACIONES = int(os.environ.get("CASOS_GEN", "120"))
SALIDA = HERE / ("casos_estudio" if GENERACIONES == 120 else "casos_estudio_prueba")
CASOS = {
    "movielens": {"usuario": "224", "prefijo": "", "item": "Ítems",
                  "evol": "fig1_evolucion_latente", "desc": "fig2_descomposicion",
                  "evid": "fig3_why_similar", "conf": "fig4_confianza_mapa",
                  "fact": "fig5_factores", "radar": "fig6_radar"},
    "bookcrossing": {"usuario": "3331", "prefijo": "Book-Crossing: ", "item": "Ítems",
                     "desc": "fig7_bx_descomposicion", "evid": "fig8_bx_evidencia",
                     "conf": "fig9_bx_confianza", "evol": "fig10_bx_evolucion_latente",
                     "fact": "fig11_bx_factores", "radar": "fig12_bx_radar"},
    "lastfm": {"usuario": "333", "prefijo": "Last.fm: ", "item": "Artistas",
               "evol": "fig13_lf_evolucion_latente", "desc": "fig14_lf_descomposicion",
               "evid": "fig15_lf_evidencia", "conf": "fig16_lf_confianza",
               "fact": "fig17_lf_factores", "radar": "fig18_lf_radar"},
}


def leer_gz(ruta):
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


def construir(ds):
    return EvolutionaryCollaborativeFilter(
        n_users=len(ds.idx_to_user), n_items=len(ds.idx_to_item), factors=K,
        population_size=25, generations=GENERACIONES, mutation_sigma=0.08,
        mutation_rate=0.02, crossover_rate=0.60, guidance_rate=0.01,
        guidance_batch=8192, bias_guidance_ratio=1.0, init_scale=0.3,
        seed=42, verbose=False,
    )


def diferencia_maxima(b, guardado):
    d = abs(b.global_mean - guardado["global_mean"])
    d = max(d, abs(b.fitness - guardado["fitness"]))
    for clave in ("user_bias", "item_bias"):
        d = max(d, max(abs(x - y) for x, y in zip(getattr(b, clave), guardado[clave])))
    for clave in ("user_factors", "item_factors"):
        for fx, fy in zip(getattr(b, clave), guardado[clave]):
            d = max(d, max(abs(x - y) for x, y in zip(fx, fy)))
    return d


def percentil(valor, lista):
    return 100.0 * sum(x <= valor for x in lista) / len(lista)


def main(nombre):
    caso = CASOS[nombre]
    out = SALIDA / nombre
    out.mkdir(parents=True, exist_ok=True)
    pre = caso["prefijo"]

    cargar, _tau = cp.CONJUNTOS[nombre]
    ds = cargar()
    split = em.split_train_test_by_user(ds.ratings, 0.20, 42)
    train_ds = em.dataset_with_ratings(ds, split.train)
    known_map = known_items_by_user(train_ds.ratings)

    guardado = json.loads((HERE / "modelos" / f"{nombre}_seed42_train.json")
                          .read_text(encoding="utf-8"))
    previo = out / "datos.json"
    if previo.exists() and os.environ.get("CASOS_REENTRENAR") != "1":
        dp = json.loads(previo.read_text(encoding="utf-8"))
        fz = dp["fitness"]
        st = [{"generation": a, "best": b_, "mean": m_, "worst": w_, "mutation_sigma": s_}
              for a, b_, m_, w_, s_ in zip(fz["generaciones"], fz["mejor"], fz["promedio"],
                                           fz["peor"], fz["sigma"])]
        seg, dif = dp["segundos_entrenamiento"], dp["diferencia_maxima_con_tablas"]
        modelo = construir(ds)
        print(f"[{nombre}] curva de fitness reutilizada de {previo.name}", flush=True)
    else:
        t = time.time()
        modelo = construir(ds).fit(split.train)
        seg = time.time() - t
        st = modelo.generation_stats_
        dif = diferencia_maxima(modelo.best_, guardado)
        print(f"[{nombre}] entrenado en {seg:.0f} s; diferencia maxima con el modelo "
              f"de las tablas = {dif:.3e}", flush=True)
    modelo.best_ = Individual(**guardado)
    b = modelo.best_

    uid = caso["usuario"]
    u = ds.user_to_idx[uid]
    conocidos = known_map.get(u, set())
    recs, corte = modelo.recommend_with_cutoff(u, conocidos, TOP_N)
    ev = build_recommendation_evidence(modelo, train_ds, u, recs)
    filas = em.evaluate_user_recommendations(modelo, u, recs, ev,
                                             next_best_raw_score=corte)
    comps = [modelo.prediction_components(u, i) for i, _ in recs]
    ancho = 26 if nombre == "movielens" else 24
    labels = [short_label(item_label(ds, i), ancho) for i, _ in recs]
    x = list(range(len(recs)))
    n_test = sum(1 for uu, _, _ in split.test if uu == u)

    pob = leer_gz(HERE / "poblacion" / nombre / "part42" / "evo_dir_s0.3_seed42.json.gz")["filas"]
    ks = [k for k, uu in enumerate(pob["usuario"]) if uu == u]
    medias_pob = {m: statistics.fmean(pob[m][k] for k in ks)
                  for m in ("confidence", "preference_match", "ablation_shift")}
    medias_caso = {"confidence": statistics.fmean(f["confidence"] for f in filas),
                   "preference_match": statistics.fmean(f["preference_match"] for f in filas),
                   "ablation_shift": statistics.fmean(f["ablation_shift"] for f in filas)}

    d = {"conjunto": nombre, "usuario": uid, "indice": u, "n_train": len(conocidos),
         "n_test": n_test, "segundos_entrenamiento": seg,
         "diferencia_maxima_con_tablas": dif,
         "medias_usuario_poblacion": medias_pob, "medias_usuario_caso": medias_caso}

    # DINÁMICA EVOLUTIVA Y ESPACIO LATENTE
    g = [s["generation"] for s in st]
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    ax = axes[0]
    ax.plot(g, [s["best"] for s in st], label="Mejor", color="#2e7d32")
    ax.plot(g, [s["mean"] for s in st], label="Promedio", color="#1565c0")
    ax.plot(g, [s["worst"] for s in st], label="Peor", color="#c62828")
    ax.set_xlabel("Generación"); ax.set_ylabel("Fitness"); ax.legend(loc="upper right")
    ax.grid(alpha=0.25); ax.set_title(f"{pre}dinámica del proceso evolutivo" if pre
                                      else "Dinámica del proceso evolutivo")
    ax2 = ax.twinx()
    ax2.plot(g, [s["mutation_sigma"] for s in st], "--", color="#e53935")
    ax2.set_ylabel("Sigma de mutación")

    ax = axes[1]
    ux, uy = two_factor_point(b.user_factors[u])
    pares = ecf.ratings_for_user(train_ds.ratings, u)
    kp = [two_factor_point(b.item_factors[i]) for i, _ in pares]
    rp = [two_factor_point(b.item_factors[i]) for i, _ in recs]
    ax.scatter([ux], [uy], marker="*", s=280, color="#e45756",
               label=f"Usuario {uid}", zorder=5)
    ax.scatter([p[0] for p in kp], [p[1] for p in kp], s=40, color="#59a14f",
               alpha=0.6, label=f"{caso['item']} conocidos ({len(kp)})")
    ax.scatter([p[0] for p in rp], [p[1] for p in rp], s=70, color="#4c78a8",
               label="Recomendaciones", zorder=4)
    for (px, py), lab in zip(rp, labels):
        ax.annotate(lab, (px, py), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("$F_1$"); ax.set_ylabel("$F_2$"); ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25); ax.set_title("Espacio latente ($F_1 \\times F_2$)")
    fig.tight_layout(); fig.savefig(out / f"{caso['evol']}.png", dpi=140); plt.close(fig)

    mejor = [s["best"] for s in st]
    mejora = mejor[0] - mejor[-1]
    gen_frac = {str(p): next(gg for gg, v in zip(g, mejor) if mejor[0] - v >= p / 100 * mejora)
                for p in (50, 80, 90, 95)} if mejora > 0 else {}
    cx = statistics.fmean(p[0] for p in kp); cy = statistics.fmean(p[1] for p in kp)
    dist_items = [math.hypot(p[0] - cx, p[1] - cy) for p in kp]
    dist_user = math.hypot(ux - cx, uy - cy)
    radios_origen = [math.hypot(*p) for p in kp]
    d["fitness"] = {"generaciones": g, "mejor": mejor, "promedio": [s["mean"] for s in st],
                    "peor": [s["worst"] for s in st],
                    "sigma": [s["mutation_sigma"] for s in st],
                    "generacion_que_alcanza_pct_de_la_mejora": gen_frac,
                    "separacion_final_mejor_peor": st[-1]["worst"] - st[-1]["best"],
                    "separacion_gen40": st[40]["worst"] - st[40]["best"] if len(st) > 40 else None}
    d["plano"] = {"usuario": [ux, uy], "centroide_conocidos": [cx, cy],
                  "distancia_usuario_al_centroide": dist_user,
                  "percentil_de_esa_distancia_entre_conocidos": percentil(dist_user, dist_items),
                  "radio_usuario_al_origen": math.hypot(ux, uy),
                  "percentil_radio_usuario_entre_radios_conocidos": percentil(math.hypot(ux, uy), radios_origen),
                  "conocidos": kp,
                  "recomendaciones": [[lab, p[0], p[1]] for lab, p in zip(labels, rp)]}

    # DESCOMPOSICIÓN DEL SCORE
    fig, ax = plt.subplots(figsize=(13, 6))
    mu0 = comps[0]["global_mean"]
    ax.axhline(mu0, color="#4c9f9f", lw=2, zorder=1,
               label=f"Media global ($\\mu$ = {mu0:.3f})")
    usados = set()
    extremos = [mu0]
    for k, c in enumerate(comps):
        arriba = abajo = c["global_mean"]
        for clave, nombre_c, color in (("user_bias", "Sesgo usuario ($b_u$)", "#f0a04b"),
                                       ("item_bias", "Sesgo ítem ($b_i$)", "#7fb069"),
                                       ("latent_dot", "Afinidad latente", "#9b6bb0")):
            v = c[clave]
            etiqueta = None if nombre_c in usados else nombre_c
            usados.add(nombre_c)
            if v >= 0:
                ax.bar(k, v, bottom=arriba, color=color, label=etiqueta, zorder=2)
                arriba += v
            else:
                ax.bar(k, v, bottom=abajo, color=color, label=etiqueta, zorder=2,
                       hatch="//", edgecolor="#333", linewidth=0.5)
                abajo += v
        extremos += [arriba, abajo, c["prediction"]]
    ax.plot(x, [c["prediction"] for c in comps], "ko-", label="Predicción final",
            ms=6, zorder=3)
    lo, hi = min(extremos), max(extremos)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"{pre}descomposición del score (usuario {uid})" if pre
                 else "Descomposición del score de predicción")
    fig.tight_layout(); fig.savefig(out / f"{caso['desc']}.png", dpi=140); plt.close(fig)

    # RAZONES Y EVIDENCIA
    w = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    ax = axes[0]
    pref = [e["preference_match"] for e in ev]
    pred = [(e["score"] - 1.0) / 4.0 for e in ev]
    pop = [e["popularity_score"] for e in ev]
    ax.bar([i - w for i in x], pref, w, label="Similitud con tus gustos", color="#4c78a8")
    ax.bar(x, pred, w, label="Predicción personalizada", color="#f0a04b")
    ax.bar([i + w for i in x], pop, w, label="Evidencia en la comunidad", color="#7fb069")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Fuerza de la evidencia"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"Razones de cada recomendación (usuario {uid})")
    ax = axes[1]
    filas_ev = [(f"{lab}\npor {short_label(item_label(ds, it), 22)}", sim, rt)
                for e, lab in zip(ev[:5], labels[:5]) for it, sim, rt in e["similar_known"][:3]]
    if filas_ev:
        col = ["#59a14f" if r[2] >= 4 else "#f0a04b" if r[2] >= 3 else "#e15759"
               for r in filas_ev]
        ax.barh([r[0] for r in filas_ev][::-1], [r[1] for r in filas_ev][::-1],
                color=col[::-1])
        ax.set_xlim(0, 1)
    else:
        ax.text(0.5, 0.5, "Sin evidencia que supere los umbrales", ha="center", fontsize=11)
    ax.set_xlabel("Similitud coseno latente"); ax.grid(axis="x", alpha=0.25)
    ax.set_title(f"{caso['item']} conocidos que respaldan las cinco primeras recomendaciones")
    fig.tight_layout(); fig.savefig(out / f"{caso['evid']}.png", dpi=140); plt.close(fig)

    # CONFIANZA Y MAPA ACCESIBLE EN MOVIELENS
    conf = [e["confidence"] for e in ev]
    bajo, alto = ecf.DEFAULT_CONFIDENCE_THRESHOLDS
    col = ["#e15759" if c < bajo else "#f0a04b" if c < alto else "#59a14f" for c in conf]
    if nombre == "movielens":
        fig, axes = plt.subplots(2, 1, figsize=(11, 11))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], conf[::-1], color=col[::-1])
    ax.set_xlim(0, 1); ax.axvline(bajo, ls="--", c="#999"); ax.axvline(alto, ls="--", c="#999")
    ax.set_xlabel("Confianza explicativa"); ax.grid(axis="x", alpha=0.25)
    ax.set_title(f"{pre}confianza explicativa (usuario {uid}, {len(conocidos)} "
                 f"interacciones de entrenamiento)" if pre
                 else "Nivel de respaldo de cada recomendación")
    if nombre == "movielens":
        ax = axes[1]
        ax.scatter([ux], [uy], marker="*", s=280, color="#e45756", label="Tu perfil", zorder=5)
        ax.scatter([p[0] for p in kp], [p[1] for p in kp], s=40, color="#59a14f",
                   alpha=0.6, label="Lo que ya conoces")
        ax.scatter([p[0] for p in rp], [p[1] for p in rp], s=70, color="#4c78a8",
                   label="Sugerencias", zorder=4)
        for (px, py), lab in zip(rp, labels):
            ax.annotate(lab, (px, py), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Dimensión de gusto 1"); ax.set_ylabel("Dimensión de gusto 2")
        ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_title("Mapa de tus preferencias")
    fig.tight_layout(); fig.savefig(out / f"{caso['conf']}.png", dpi=140); plt.close(fig)

    # CONTRIBUCIÓN POR FACTOR
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=True)
    for ax, c, lab in zip(axes, comps[:5], labels[:5]):
        v = c["factor_contributions"]
        ax.bar(range(1, K + 1), v, color=["#59a14f" if z >= 0 else "#e15759" for z in v])
        ax.set_title(lab, fontsize=8); ax.set_xticks(range(1, K + 1))
        ax.set_xticklabels([f"$F_{j}$" for j in range(1, K + 1)], fontsize=7)
        ax.axhline(0, color="#333", lw=0.8); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Contribución al score")
    fig.suptitle(f"{pre}contribución de cada factor latente" if pre
                 else "Contribución de cada factor latente")
    fig.tight_layout(); fig.savefig(out / f"{caso['fact']}.png", dpi=140); plt.close(fig)

    # RADAR
    def norm(v):
        m = max(abs(z) for z in v) or 1.0
        return [(z / m + 1) / 2 for z in v]

    uvec, ivec = norm(b.user_factors[u]), norm(b.item_factors[recs[0][0]])
    ang = [n / K * 2 * math.pi for n in range(K)] + [0]
    fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(111, polar=True)
    ax.plot(ang, uvec + uvec[:1], color="#4c78a8", lw=2, label=f"Usuario {uid}")
    ax.fill(ang, uvec + uvec[:1], color="#4c78a8", alpha=0.25)
    ax.plot(ang, ivec + ivec[:1], color="#f0a04b", lw=2, label=labels[0])
    ax.fill(ang, ivec + ivec[:1], color="#f0a04b", alpha=0.25)
    t_ = [2 * math.pi * j / 200 for j in range(201)]
    ax.plot(t_, [0.5] * len(t_), "--", color="#555", lw=1.2, label="Cero (dentro: negativo)")
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["−máx", "0", "+máx"], fontsize=8)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([f"$F_{j + 1}$" for j in range(K)])
    ax.set_title(f"{pre}perfil latente usuario frente a ítem" if pre
                 else "Perfil latente: usuario frente a ítem recomendado")
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"{caso['radar']}.png", dpi=140); plt.close(fig)

    d["radar"] = {"item": labels[0], "usuario_norm": uvec, "item_norm": ivec,
                  "usuario_bruto": b.user_factors[u], "item_bruto": b.item_factors[recs[0][0]],
                  "coseno": cosine_similarity(b.user_factors[u], b.item_factors[recs[0][0]])}
    d["recomendaciones"] = []
    for lab, (i, s), c, e, f in zip(labels, recs, comps, ev, filas):
        d["recomendaciones"].append({
            "item": lab, "id": ds.idx_to_item[i], "titulo": item_label(ds, i),
            "score": s, "prediccion": c["prediction"], "mu": c["global_mean"],
            "b_u": c["user_bias"], "b_i": c["item_bias"], "latente": c["latent_dot"],
            "factores": c["factor_contributions"],
            "preference_match": e["preference_match"], "popularidad": e["popularity_score"],
            "confianza": e["confidence"], "senales_confianza": e["confidence_signals"],
            "n_evidencia": len(e["similar_known"]),
            "evidencia": [[item_label(ds, it), sim, rt] for it, sim, rt in e["similar_known"]],
            "ablation_shift": f["ablation_shift"], "necesidad": f["necessity"],
            "suficiencia": f["sufficiency"], "firma_factores": list(f["top_factor_signature"])})
    (out / "datos.json").write_text(json.dumps(d, indent=1, ensure_ascii=False),
                                    encoding="utf-8")
    print(f"[{nombre}] usuario {uid}: n_train={len(conocidos)} n_test={n_test}; "
          f"medias caso {medias_caso} frente a poblacion {medias_pob}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CASOS:
        raise SystemExit(__doc__)
    main(sys.argv[1])
