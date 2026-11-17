"""
Métricas offline de calidad explicativa y de calidad del recomendador.

Uso: python explanation_metrics.py --movielens-100k datasets/ml-100k --evaluate-stability
     python explanation_metrics.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from evolutionary_cf import (
    DEFAULT_MIN_EVIDENCE_RATING,
    DEFAULT_MIN_EVIDENCE_SIMILARITY,
    Dataset,
    EvolutionaryCollaborativeFilter,
    Rating,
    build_recommendation_evidence,
    filter_dataset,
    known_items_by_user,
    load_bookcrossing_folder,
    load_interactions_csv,
    load_item_titles_csv,
    load_lastfm_hetrec_folder,
    load_movielens_100k,
)


Recommendation = Tuple[int, float]

# CONSTANTES
DOMINANT_FACTORS = 3

LATENT_RELEVANCE_RATIO = 0.01


# DATOS Y MODELO
@dataclass(frozen=True)
class RatingSplit:
    train: List[Rating]
    test: List[Rating]


def load_dataset_from_args(args: argparse.Namespace) -> Dataset:
    if args.dataset == "bookcrossing":
        if args.data_dir:
            return load_bookcrossing_folder(
                Path(args.data_dir),
                min_user_interactions=args.min_user_interactions or 5,
                min_item_interactions=args.min_item_interactions or 5,
            )
        return load_interactions_csv(
            Path(args.csv),
            user_column=args.user_col or "User-ID",
            item_column=args.item_col or "ISBN",
            rating_column=args.rating_col or "Rating",
            delimiter=args.delimiter or ";",
            encoding=args.encoding,
            drop_zero_ratings=True,
            rating_transform=args.rating_transform or "scale-1-5",
            item_titles_by_external_id=load_optional_item_titles(args),
        )

    if args.dataset == "lastfm":
        if args.data_dir:
            return load_lastfm_hetrec_folder(
                Path(args.data_dir),
                min_user_interactions=args.min_user_interactions or 5,
                min_item_interactions=args.min_item_interactions or 5,
            )
        return load_interactions_csv(
            Path(args.csv),
            user_column=args.user_col or "userID",
            item_column=args.item_col or "artistID",
            rating_column=args.rating_col or "weight",
            delimiter=args.delimiter or ",",
            encoding=args.encoding,
            drop_zero_ratings=False,
            rating_transform=args.rating_transform or "log-scale-1-5",
            item_titles_by_external_id=load_optional_item_titles(args),
        )

    if args.movielens_100k:
        return load_movielens_100k(Path(args.movielens_100k))

    if not args.csv:
        raise ValueError("--csv, --data-dir, or --movielens-100k is required")

    dataset = load_interactions_csv(
        Path(args.csv),
        user_column=args.user_col or "user_id",
        item_column=args.item_col or "item_id",
        rating_column=args.rating_col or "rating",
        delimiter=args.delimiter or ",",
        encoding=args.encoding,
        drop_zero_ratings=bool(args.drop_zero_ratings),
        rating_transform=args.rating_transform or "none",
        item_titles_by_external_id=load_optional_item_titles(args),
    )
    return filter_dataset(
        dataset,
        min_user_interactions=args.min_user_interactions or 1,
        min_item_interactions=args.min_item_interactions or 1,
    )


def load_optional_item_titles(args: argparse.Namespace) -> Dict[str, str] | None:
    if not args.items_csv:
        return None
    return load_item_titles_csv(
        Path(args.items_csv),
        item_id_column=args.item_id_col or args.item_col or "item_id",
        item_name_column=args.item_name_col or "title",
        delimiter=args.items_delimiter or args.delimiter or ",",
        encoding=args.items_encoding or args.encoding,
    )


def train_model(
    args: argparse.Namespace,
    dataset: Dataset,
    seed: int,
    ratings: Sequence[Rating] | None = None,
) -> EvolutionaryCollaborativeFilter:
    model = EvolutionaryCollaborativeFilter(
        n_users=len(dataset.idx_to_user),
        n_items=len(dataset.idx_to_item),
        factors=args.factors,
        population_size=args.population,
        generations=args.generations,
        mutation_sigma=args.mutation_sigma,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        guidance_rate=args.guidance_rate,
        guidance_batch=args.guidance_batch,
        init_scale=args.init_scale,
        seed=seed,
        verbose=args.verbose,
    )
    return model.fit(ratings or dataset.ratings)


def dataset_with_ratings(dataset: Dataset, ratings: Sequence[Rating]) -> Dataset:
    return Dataset(
        ratings=list(ratings),
        user_to_idx=dataset.user_to_idx,
        item_to_idx=dataset.item_to_idx,
        idx_to_user=dataset.idx_to_user,
        idx_to_item=dataset.idx_to_item,
        item_titles=dataset.item_titles,
    )


# PARTICIÓN Y USUARIOS EVALUADOS
def split_train_test_by_user(
    ratings: Sequence[Rating],
    test_ratio: float,
    seed: int,
) -> RatingSplit:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    by_user: Dict[int, List[Rating]] = {}
    for rating in ratings:
        by_user.setdefault(rating[0], []).append(rating)

    rng = random.Random(seed)
    train: List[Rating] = []
    test: List[Rating] = []
    for user_ratings in by_user.values():
        shuffled = user_ratings[:]
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train.extend(shuffled)
            continue

        test_count = max(1, int(round(len(shuffled) * test_ratio)))
        test_count = min(test_count, len(shuffled) - 1)
        test.extend(shuffled[:test_count])
        train.extend(shuffled[test_count:])

    if not train or not test:
        raise ValueError("Could not create non-empty train/test split")
    return RatingSplit(train=train, test=test)


def sample_user_indices(
    dataset: Dataset,
    sample_users: int,
    top_n: int,
    seed: int,
    train_ratings: Sequence[Rating] | None = None,
    test_ratings: Sequence[Rating] | None = None,
    relevant_threshold: float = 4.0,
) -> List[int]:
    known = known_items_by_user(train_ratings or dataset.ratings)
    relevant_test = relevant_items_by_user(test_ratings or [], relevant_threshold)
    candidates = [
        user_idx
        for user_idx, items in known.items()
        if len(items) > 0
        and len(items) + top_n <= len(dataset.idx_to_item)
        and (not test_ratings or relevant_test.get(user_idx))
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(sample_users, len(candidates))]


def relevant_items_by_user(
    ratings: Sequence[Rating],
    relevant_threshold: float,
) -> Dict[int, set[int]]:
    relevant: Dict[int, set[int]] = {}
    for user_idx, item_idx, rating in ratings:
        if rating >= relevant_threshold:
            relevant.setdefault(user_idx, set()).add(item_idx)
    return relevant


def recommendations_for_user(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    top_n: int,
    known_map: Dict[int, set[int]] | None = None,
) -> List[Recommendation]:
    known = known_map if known_map is not None else known_items_by_user(dataset.ratings)
    return model.recommend(user_idx, known.get(user_idx, set()), top_n)


# FIDELIDAD POR ABLACIÓN

def raw_prediction_excluding_factors(
    model: EvolutionaryCollaborativeFilter,
    user_idx: int,
    item_idx: int,
    excluded: Iterable[int],
) -> float:
    """Predicción cruda (sin recorte) al anular las dimensiones indicadas."""
    if model.best_ is None:
        raise RuntimeError("fit must be called before raw_prediction_excluding_factors")

    excluded_set = set(excluded)
    user_vector = model.best_.user_factors[user_idx]
    item_vector = model.best_.item_factors[item_idx]
    dot = sum(
        user_value * item_value
        for position, (user_value, item_value) in enumerate(zip(user_vector, item_vector))
        if position not in excluded_set
    )
    return (
        model.best_.global_mean
        + model.best_.user_bias[user_idx]
        + model.best_.item_bias[item_idx]
        + dot
    )


# MÉTRICAS POR RECOMENDACIÓN
def evaluate_explanations(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_indices: Sequence[int],
    top_n: int,
    min_evidence_similarity: float = DEFAULT_MIN_EVIDENCE_SIMILARITY,
    min_evidence_rating: float = DEFAULT_MIN_EVIDENCE_RATING,
) -> Dict[str, Any]:
    rows = collect_explanation_rows(
        model,
        dataset,
        user_indices,
        top_n,
        min_evidence_similarity=min_evidence_similarity,
        min_evidence_rating=min_evidence_rating,
    )
    return aggregate_rows(rows)


def collect_explanation_rows(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_indices: Sequence[int],
    top_n: int,
    min_evidence_similarity: float = DEFAULT_MIN_EVIDENCE_SIMILARITY,
    min_evidence_rating: float = DEFAULT_MIN_EVIDENCE_RATING,
    known_map: Dict[int, set[int]] | None = None,
) -> List[Dict[str, Any]]:
    """Una fila de métricas por recomendación."""
    known_map = known_map if known_map is not None else known_items_by_user(dataset.ratings)
    rows: List[Dict[str, Any]] = []
    for user_idx in user_indices:
        recommendations, cutoff = model.recommend_with_cutoff(
            user_idx, known_map.get(user_idx, set()), top_n
        )
        evidence = build_recommendation_evidence(
            model,
            dataset,
            user_idx,
            recommendations,
            min_similarity=min_evidence_similarity,
            min_rating=min_evidence_rating,
        )
        rows.extend(
            evaluate_user_recommendations(
                model,
                user_idx,
                recommendations,
                evidence,
                next_best_raw_score=cutoff,
            )
        )
    return rows


def evaluate_user_recommendations(
    model: EvolutionaryCollaborativeFilter,
    user_idx: int,
    recommendations: Sequence[Recommendation],
    evidence: Sequence[Dict[str, Any]],
    next_best_raw_score: float | None = None,
) -> List[Dict[str, Any]]:
    rows = []
    rating_range = max(model.rating_max - model.rating_min, 1e-12)
    cutoff_score = next_best_raw_score

    for recommendation, entry in zip(recommendations, evidence):
        item_idx, score = recommendation
        components = entry["components"]
        factor_contributions = components["factor_contributions"]

        # VERIFICACIÓN DE EXACTITUD
        reconstructed_raw = (
            components["global_mean"]
            + components["user_bias"]
            + components["item_bias"]
            + sum(factor_contributions)
        )
        reconstruction_error = abs(reconstructed_raw - components["raw_prediction"])

        # ABLACIÓN DE LOS FACTORES DOMINANTES
        dominant = top_abs_indices(factor_contributions, DOMINANT_FACTORS)
        ablated_raw = raw_prediction_excluding_factors(model, user_idx, item_idx, dominant)
        ablation_shift = abs(components["raw_prediction"] - ablated_raw) / rating_range
        necessity = (
            1.0 if cutoff_score is not None and ablated_raw < cutoff_score else 0.0
        )
        no_dominantes = [
            j for j in range(len(factor_contributions)) if j not in set(dominant)
        ]
        solo_dominantes_raw = raw_prediction_excluding_factors(
            model, user_idx, item_idx, no_dominantes
        )
        sufficiency = (
            1.0 if cutoff_score is not None and solo_dominantes_raw >= cutoff_score else 0.0
        )

        # COBERTURA CON UMBRALES
        latent_is_relevant = abs(components["latent_dot"]) >= LATENT_RELEVANCE_RATIO * rating_range
        has_evidence = bool(entry["similar_known"])

        rows.append(
            {
                "user_idx": float(user_idx),
                "item_idx": float(item_idx),
                "score": score,
                "raw_score": components["raw_prediction"],
                "reconstruction_error": reconstruction_error,
                "ablation_shift": ablation_shift,
                "necessity": necessity,
                "sufficiency": sufficiency,
                "intrinsic_explanation": 1.0 if latent_is_relevant else 0.0,
                "user_evidence_explanation": 1.0 if has_evidence else 0.0,
                "coverage": 1.0 if (latent_is_relevant or has_evidence) else 0.0,
                "confidence": entry["confidence"],
                "preference_match": entry["preference_match"],
                "similar_evidence_count": float(len(entry["similar_known"])),
                "top_factor_signature": tuple(dominant),
                "similar_item_signature": tuple(
                    item for item, _, _ in entry["similar_known"][:3]
                ),
            }
        )
    return rows


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "recommendations_evaluated": 0,
            "reconstruction_max_error": None,
            "reconstruction_exact": None,
            "mean_ablation_shift": None,
            "necessity": None,
            "sufficiency": None,
            "coverage": None,
            "intrinsic_coverage": None,
            "user_evidence_coverage": None,
            "mean_confidence": None,
            "mean_preference_match": None,
            "mean_similar_evidence_count": None,
            "score_preference_correlation": None,
            "score_preference_correlation_by_user": None,
        }

    max_reconstruction_error = max(row["reconstruction_error"] for row in rows)
    return {
        "recommendations_evaluated": len(rows),
        "reconstruction_max_error": max_reconstruction_error,
        "reconstruction_exact": max_reconstruction_error < 1e-9,
        "mean_ablation_shift": mean(row["ablation_shift"] for row in rows),
        "necessity": mean(row["necessity"] for row in rows),
        "sufficiency": mean(row["sufficiency"] for row in rows),
        "coverage": mean(row["coverage"] for row in rows),
        "intrinsic_coverage": mean(row["intrinsic_explanation"] for row in rows),
        "user_evidence_coverage": mean(row["user_evidence_explanation"] for row in rows),
        "mean_confidence": mean(row["confidence"] for row in rows),
        "mean_preference_match": mean(row["preference_match"] for row in rows),
        "mean_similar_evidence_count": mean(row["similar_evidence_count"] for row in rows),
        "score_preference_correlation": pearson(
            [row["raw_score"] for row in rows],
            [row["preference_match"] for row in rows],
        ),
        "score_preference_correlation_by_user": mean_within_user_correlation(rows),
    }


def mean_within_user_correlation(rows: Sequence[Dict[str, Any]]) -> float | None:
    by_user: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        by_user.setdefault(row["user_idx"], []).append(row)

    correlations = []
    for user_rows in by_user.values():
        if len(user_rows) < 3:
            continue
        value = pearson(
            [row["raw_score"] for row in user_rows],
            [row["preference_match"] for row in user_rows],
        )
        if value is not None:
            correlations.append(value)
    return mean(correlations) if correlations else None


# ESTABILIDAD ENTRE DOS MODELOS
def evaluate_stability(
    model_a: EvolutionaryCollaborativeFilter,
    model_b: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_indices: Sequence[int],
    top_n: int,
    min_evidence_similarity: float = DEFAULT_MIN_EVIDENCE_SIMILARITY,
    min_evidence_rating: float = DEFAULT_MIN_EVIDENCE_RATING,
) -> Dict[str, float]:
    known_map = known_items_by_user(dataset.ratings)
    recommendation_jaccards = []
    factor_jaccards = []
    similar_item_jaccards = []

    for user_idx in user_indices:
        conocidos = known_map.get(user_idx, set())
        recs_a, cutoff_a = model_a.recommend_with_cutoff(user_idx, conocidos, top_n)
        recs_b, cutoff_b = model_b.recommend_with_cutoff(user_idx, conocidos, top_n)
        evidence_a = build_recommendation_evidence(
            model_a, dataset, user_idx, recs_a,
            min_similarity=min_evidence_similarity, min_rating=min_evidence_rating,
        )
        evidence_b = build_recommendation_evidence(
            model_b, dataset, user_idx, recs_b,
            min_similarity=min_evidence_similarity, min_rating=min_evidence_rating,
        )
        rows_a = evaluate_user_recommendations(
            model_a, user_idx, recs_a, evidence_a, next_best_raw_score=cutoff_a)
        rows_b = evaluate_user_recommendations(
            model_b, user_idx, recs_b, evidence_b, next_best_raw_score=cutoff_b)

        rec_set_a = {item_idx for item_idx, _ in recs_a}
        rec_set_b = {item_idx for item_idx, _ in recs_b}
        recommendation_jaccards.append(jaccard(rec_set_a, rec_set_b))

        by_item_b = {int(row["item_idx"]): row for row in rows_b}
        for row_a in rows_a:
            item_idx = int(row_a["item_idx"])
            if item_idx not in by_item_b:
                continue
            row_b = by_item_b[item_idx]
            factor_jaccards.append(
                jaccard(set(row_a["top_factor_signature"]), set(row_b["top_factor_signature"]))
            )
            sig_a = set(row_a["similar_item_signature"])
            sig_b = set(row_b["similar_item_signature"])
            if sig_a or sig_b:
                similar_item_jaccards.append(jaccard(sig_a, sig_b))

    recommendation_jaccard = mean(recommendation_jaccards) if recommendation_jaccards else 0.0
    top_factor_jaccard = mean(factor_jaccards) if factor_jaccards else 0.0
    similar_item_jaccard = mean(similar_item_jaccards) if similar_item_jaccards else 0.0
    return {
        "recommendation_jaccard": recommendation_jaccard,
        "top_factor_jaccard": top_factor_jaccard,
        "similar_item_jaccard": similar_item_jaccard,
    }


# TOLERANCIA A LA ESCASEZ
def evaluate_sparsity_tolerance(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_indices: Sequence[int],
    top_n: int,
    min_evidence_similarity: float = DEFAULT_MIN_EVIDENCE_SIMILARITY,
    min_evidence_rating: float = DEFAULT_MIN_EVIDENCE_RATING,
    rows: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    known = known_items_by_user(dataset.ratings)
    if rows is None:
        rows = collect_explanation_rows(
            model,
            dataset,
            user_indices,
            top_n,
            min_evidence_similarity=min_evidence_similarity,
            min_evidence_rating=min_evidence_rating,
            known_map=known,
        )

    buckets: Dict[str, List[int]] = {
        "sparse_1_5": [],
        "medium_6_20": [],
        "dense_21_plus": [],
    }
    for user_idx in user_indices:
        count = len(known.get(user_idx, set()))
        if count <= 5:
            buckets["sparse_1_5"].append(user_idx)
        elif count <= 20:
            buckets["medium_6_20"].append(user_idx)
        else:
            buckets["dense_21_plus"].append(user_idx)

    result: Dict[str, Any] = {}
    for name, indices in buckets.items():
        index_set = {float(user_idx) for user_idx in indices}
        bucket_rows = [row for row in rows if row["user_idx"] in index_set]
        result[name] = {"users": len(indices), **aggregate_rows(bucket_rows)}
    return result


# CALIDAD DEL RECOMENDADOR
def evaluate_recommender_quality(
    model: EvolutionaryCollaborativeFilter,
    train_dataset: Dataset,
    test_ratings: Sequence[Rating],
    user_indices: Sequence[int],
    top_n: int,
    relevant_threshold: float,
) -> Dict[str, Any]:
    squared_errors = []
    absolute_errors = []
    for user_idx, item_idx, rating in test_ratings:
        error = rating - model.predict(user_idx, item_idx)
        squared_errors.append(error * error)
        absolute_errors.append(abs(error))

    relevant_by_user = relevant_items_by_user(test_ratings, relevant_threshold)
    train_known = known_items_by_user(train_dataset.ratings)
    precision_values = []
    recall_values = []
    ndcg_values = []
    hit_values = []
    recommended_items: set[int] = set()
    diversity_values = []

    for user_idx in user_indices:
        relevant = relevant_by_user.get(user_idx, set())
        if not relevant:
            continue

        recommendations = model.recommend(user_idx, train_known.get(user_idx, set()), top_n)
        recommended = [item_idx for item_idx, _ in recommendations]
        recommended_items.update(recommended)

        hits = len(set(recommended) & relevant)
        precision_values.append(hits / max(1, top_n))
        recall_values.append(hits / len(relevant))
        hit_values.append(1.0 if hits > 0 else 0.0)
        ndcg_values.append(ndcg_at_k(recommended, relevant, top_n))

        diversity = intra_list_diversity(model, recommended)
        if diversity is not None:
            diversity_values.append(diversity)

    return {
        "test_ratings": float(len(test_ratings)),
        "rmse": math.sqrt(mean(squared_errors)) if squared_errors else None,
        "mae": mean(absolute_errors) if absolute_errors else None,
        f"precision_at_{top_n}": mean(precision_values) if precision_values else None,
        f"recall_at_{top_n}": mean(recall_values) if recall_values else None,
        f"ndcg_at_{top_n}": mean(ndcg_values) if ndcg_values else None,
        f"hit_rate_at_{top_n}": mean(hit_values) if hit_values else None,
        "ranking_users_evaluated": float(len(precision_values)),
        "catalog_coverage": len(recommended_items) / max(1, len(train_dataset.idx_to_item)),
        "intra_list_diversity": mean(diversity_values) if diversity_values else None,
    }


def ndcg_at_k(
    recommended: Sequence[int],
    relevant: set[int],
    top_k: int,
) -> float:
    """NDCG@k con relevancia binaria."""
    dcg = 0.0
    for position, item_idx in enumerate(recommended[:top_k]):
        if item_idx in relevant:
            dcg += 1.0 / math.log2(position + 2)

    ideal_hits = min(len(relevant), top_k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def intra_list_diversity(
    model: EvolutionaryCollaborativeFilter,
    item_indices: Sequence[int],
) -> float | None:
    """Distancia coseno media entre pares de la lista."""
    if model.best_ is None or len(item_indices) < 2:
        return None

    distances = []
    for left_pos, left_item in enumerate(item_indices):
        for right_item in item_indices[left_pos + 1:]:
            left = model.best_.item_factors[left_item]
            right = model.best_.item_factors[right_item]
            similarity = vector_cosine_similarity(left, right)
            normalized_similarity = (similarity + 1.0) / 2.0
            distances.append(1.0 - normalized_similarity)
    return mean(distances)


# UTILIDADES
def vector_cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def clip(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def top_abs_indices(values: Sequence[float], top_k: int) -> List[int]:
    return [
        idx
        for idx, _ in sorted(
            enumerate(values),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )[:top_k]
    ]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Correlación de Pearson, o None si no está definida."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_den = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_den = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_den == 0 or right_den == 0:
        return None
    return numerator / (left_den * right_den)


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


# LÍNEA DE COMANDOS
def run(args: argparse.Namespace) -> None:
    dataset = load_dataset_from_args(args)
    split = split_train_test_by_user(dataset.ratings, args.test_ratio, args.seed)
    train_dataset = dataset_with_ratings(dataset, split.train)
    user_indices = sample_user_indices(
        dataset,
        args.sample_users or len(dataset.idx_to_user),
        args.top_n,
        args.seed,
        train_ratings=split.train,
    )
    if not user_indices:
        raise ValueError("No users available for evaluation after filtering")

    model = train_model(args, dataset, args.seed, ratings=split.train)

    rows = collect_explanation_rows(
        model,
        train_dataset,
        user_indices,
        args.top_n,
        min_evidence_similarity=args.min_evidence_similarity,
        min_evidence_rating=args.min_evidence_rating,
    )
    quality = aggregate_rows(rows)
    sparsity = evaluate_sparsity_tolerance(
        model,
        train_dataset,
        user_indices,
        args.top_n,
        min_evidence_similarity=args.min_evidence_similarity,
        min_evidence_rating=args.min_evidence_rating,
        rows=rows,
    )
    recommender_quality = evaluate_recommender_quality(
        model,
        train_dataset,
        split.test,
        user_indices,
        args.top_n,
        args.relevant_threshold,
    )

    stability = None
    if args.evaluate_stability:
        stable_model = train_model(
            args,
            dataset,
            args.seed + args.stability_seed_delta,
            ratings=split.train,
        )
        stability = evaluate_stability(
            model,
            stable_model,
            train_dataset,
            user_indices,
            args.top_n,
            min_evidence_similarity=args.min_evidence_similarity,
            min_evidence_rating=args.min_evidence_rating,
        )

    result = {
        "dataset": args.dataset,
        "users": len(dataset.idx_to_user),
        "items": len(dataset.idx_to_item),
        "ratings": len(dataset.ratings),
        "train_ratings": len(split.train),
        "test_ratings": len(split.test),
        "test_ratio": args.test_ratio,
        "sample_users": len(user_indices),
        "top_n": args.top_n,
        "recommender_quality": recommender_quality,
        "quality": quality,
        "sparsity_tolerance": sparsity,
    }
    if stability is not None:
        result["stability"] = stability

    print_metrics(result)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nMetricas guardadas en: {output_path}")


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "---"
    if isinstance(value, bool):
        return "si" if value else "NO"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_metrics(result: Dict[str, Any]) -> None:
    top_n = result["top_n"]
    print("Resumen del dataset")
    print(f"- Usuarios: {result['users']}")
    print(f"- Items: {result['items']}")
    print(f"- Ratings/interacciones: {result['ratings']}")
    print(f"- Train ratings: {result['train_ratings']}")
    print(f"- Test ratings: {result['test_ratings']}")
    print(f"- Test ratio: {result['test_ratio']:.2f}")
    print(f"- Usuarios evaluados: {result['sample_users']}")
    print(f"- Top-N: {top_n}")

    recommender = result["recommender_quality"]
    print("\nCalidad del recomendador")
    print(f"- RMSE test: {format_value(recommender['rmse'])}")
    print(f"- MAE test: {format_value(recommender['mae'])}")
    print(f"- Precision@{top_n}: {format_value(recommender[f'precision_at_{top_n}'])}")
    print(f"- Recall@{top_n}: {format_value(recommender[f'recall_at_{top_n}'])}")
    print(f"- NDCG@{top_n}: {format_value(recommender[f'ndcg_at_{top_n}'])}")
    print(f"- Hit rate@{top_n}: {format_value(recommender[f'hit_rate_at_{top_n}'])}")
    print(f"- Usuarios ranking evaluados: {int(recommender['ranking_users_evaluated'])}")
    print(f"- Cobertura catalogo: {format_value(recommender['catalog_coverage'])}")
    print(f"- Diversidad intra-lista: {format_value(recommender['intra_list_diversity'])}")

    quality = result["quality"]
    print("\nVerificacion de exactitud de la descomposicion")
    print(f"- Error maximo de reconstruccion: {format_value(quality['reconstruction_max_error'], 2)}")
    print(f"- Descomposicion exacta: {format_value(quality['reconstruction_exact'])}")
    print("  (es una comprobacion de correccion algebraica, no una metrica de calidad)")

    print("\nCalidad explicativa")
    print(f"- Ablation shift medio (fidelidad): {format_value(quality['mean_ablation_shift'])}")
    print(f"- Necesidad (cae del Top-N tras ablacion): {format_value(quality['necessity'])}")
    print(f"- Suficiencia (se sostiene solo con los dominantes): {format_value(quality['sufficiency'])}")
    print(f"- Coverage total: {format_value(quality['coverage'])}")
    print(f"- Coverage intrinseca: {format_value(quality['intrinsic_coverage'])}")
    print(f"- Coverage evidencia usuario: {format_value(quality['user_evidence_coverage'])}")
    print(f"- Evidencia media por recomendacion: {format_value(quality['mean_similar_evidence_count'], 2)}")
    print(f"- Confianza explicativa media: {format_value(quality['mean_confidence'])}")
    print(f"- Preference match medio: {format_value(quality['mean_preference_match'])}")
    print(f"- Correlacion score/preference (global): {format_value(quality['score_preference_correlation'])}")
    print(f"- Correlacion score/preference (intra-usuario): {format_value(quality['score_preference_correlation_by_user'])}")

    if "stability" in result:
        stability = result["stability"]
        print("\nEstabilidad entre dos semillas")
        print(f"- Jaccard recomendaciones: {format_value(stability['recommendation_jaccard'])}")
        print(f"- Jaccard top factores: {format_value(stability['top_factor_jaccard'])}")
        print(f"- Jaccard items similares: {format_value(stability['similar_item_jaccard'])}")

    print("\nTolerancia a la escasez")
    for name, values in result["sparsity_tolerance"].items():
        print(
            f"- {name}: users={values['users']} "
            f"recs={values['recommendations_evaluated']} "
            f"coverage={format_value(values['coverage'])} "
            f"ablation_shift={format_value(values['mean_ablation_shift'])} "
            f"confidence={format_value(values['mean_confidence'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua metricas offline de explicabilidad para el recomendador evolutivo."
    )
    parser.add_argument("--dataset", choices=["generic", "bookcrossing", "lastfm"], default="generic")
    parser.add_argument("--csv")
    parser.add_argument("--data-dir")
    parser.add_argument("--movielens-100k")
    parser.add_argument("--user-col")
    parser.add_argument("--item-col")
    parser.add_argument("--rating-col")
    parser.add_argument("--delimiter")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--drop-zero-ratings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rating-transform", choices=["none", "scale-1-5", "log-scale-1-5"])
    parser.add_argument("--items-csv")
    parser.add_argument("--item-id-col")
    parser.add_argument("--item-name-col")
    parser.add_argument("--items-delimiter")
    parser.add_argument("--items-encoding")
    parser.add_argument("--min-user-interactions", type=int)
    parser.add_argument("--min-item-interactions", type=int)
    parser.add_argument("--sample-users", type=int, help="Por defecto, todos los usuarios")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--relevant-threshold", type=float, default=4.0)
    parser.add_argument(
        "--min-evidence-similarity",
        type=float,
        default=DEFAULT_MIN_EVIDENCE_SIMILARITY,
        help="Similitud coseno minima para que un item conocido sirva como evidencia",
    )
    parser.add_argument(
        "--min-evidence-rating",
        type=float,
        default=DEFAULT_MIN_EVIDENCE_RATING,
        help="Rating minimo para que un item conocido sirva como evidencia de apoyo",
    )
    parser.add_argument("--factors", type=int, default=6)
    parser.add_argument("--population", type=int, default=25)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--mutation-sigma", type=float, default=0.08)
    parser.add_argument("--mutation-rate", type=float, default=0.02)
    parser.add_argument("--crossover-rate", type=float, default=0.60)
    parser.add_argument(
        "--guidance-rate",
        type=float,
        default=0.01,
        help="Intensidad del paso de mutacion dirigido por residuo (0 desactiva la guia)",
    )
    parser.add_argument("--guidance-batch", type=int, default=8192)
    parser.add_argument("--init-scale", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--evaluate-stability", action="store_true")
    parser.add_argument("--stability-seed-delta", type=int, default=1)
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
