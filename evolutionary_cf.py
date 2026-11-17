"""
Recomendador por factorización matricial ajustada con un algoritmo evolutivo híbrido,
con explicaciones por descomposición del score y por evidencia del historial.

Uso: python evolutionary_cf.py --help
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Rating = Tuple[int, int, float]

# CONSTANTES DE LA EVIDENCIA Y DE LA CONFIANZA
DEFAULT_MIN_EVIDENCE_SIMILARITY = 0.0
DEFAULT_MIN_EVIDENCE_RATING = 3.0
DEFAULT_CONFIDENCE_WEIGHTS = (0.45, 0.30, 0.25)
DEFAULT_CONFIDENCE_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)
CONFIDENCE_ACTIVITY_REFERENCE = 50


# ESTRUCTURAS DE DATOS
@dataclass(frozen=True)
class Dataset:
    ratings: List[Rating]
    user_to_idx: Dict[str, int]
    item_to_idx: Dict[str, int]
    idx_to_user: List[str]
    idx_to_item: List[str]
    item_titles: Dict[int, str] | None = None


@dataclass
class Individual:
    user_factors: List[List[float]]
    item_factors: List[List[float]]
    user_bias: List[float]
    item_bias: List[float]
    global_mean: float
    fitness: float = float("inf")


# MODELO
class EvolutionaryCollaborativeFilter:
    """Factorización matricial ajustada con un algoritmo evolutivo híbrido."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int = 6,
        population_size: int = 25,
        generations: int = 120,
        mutation_sigma: float = 0.08,
        mutation_rate: float = 0.02,
        crossover_rate: float = 0.60,
        elite_ratio: float = 0.25,
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        regularization: float = 0.001,
        guidance_rate: float = 0.01,
        guidance_batch: int = 8192,
        bias_guidance_ratio: float = 1.0,
        init_scale: float | None = None,
        explainability_weight: float = 0.0,
        explainability_sample: int = 8,
        explainability_top_n: int = 10,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be positive")
        if factors <= 0:
            raise ValueError("factors must be positive")
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        if not 0 < elite_ratio < 1:
            raise ValueError("elite_ratio must be between 0 and 1")
        if not 0 < mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("crossover_rate must be between 0 and 1")
        if guidance_rate < 0:
            raise ValueError("guidance_rate must be non-negative")
        if guidance_batch <= 0:
            raise ValueError("guidance_batch must be positive")
        if init_scale is not None and init_scale <= 0:
            raise ValueError("init_scale must be positive")

        self.n_users = n_users
        self.n_items = n_items
        self.factors = factors
        self.population_size = population_size
        self.generations = generations
        self.mutation_sigma = mutation_sigma
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.regularization = regularization
        self.guidance_rate = guidance_rate
        self.guidance_batch = guidance_batch
        self.bias_guidance_ratio = bias_guidance_ratio
        self.init_scale = init_scale if init_scale is not None else 0.3
        self.explainability_weight = explainability_weight
        self.explainability_sample = explainability_sample
        self.explainability_top_n = explainability_top_n
        self._explain_users: List[int] = []
        self._known_by_user: Dict[int, Dict[int, float]] = {}
        self.random = random.Random(seed)
        self._explain_random = random.Random(seed + 10_000)
        self.verbose = verbose
        self.best_: Individual | None = None
        self.history_: List[float] = []
        self.generation_stats_: List[Dict[str, float]] = []

    def fit(self, ratings: Sequence[Rating]) -> "EvolutionaryCollaborativeFilter":
        if not ratings:
            raise ValueError("ratings cannot be empty")

        global_mean, user_bias_seed, item_bias_seed = self._baseline_terms(ratings)

        if self.explainability_weight > 0.0:
            self._known_by_user = {}
            for user_idx, item_idx, rating in ratings:
                self._known_by_user.setdefault(user_idx, {})[item_idx] = rating
            con_historial = [u for u, v in self._known_by_user.items() if v]
            con_historial.sort()
            self._explain_users = self._explain_random.sample(
                con_historial, min(self.explainability_sample, len(con_historial))
            )

        population = [
            self._new_individual(global_mean, user_bias_seed, item_bias_seed)
            for _ in range(self.population_size)
        ]

        for generation in range(self.generations):
            for individual in population:
                individual.fitness = self._fitness(individual, ratings)

            population.sort(key=lambda candidate: candidate.fitness)
            self.best_ = self._clone(population[0])
            self._record_generation(population, generation)
            if self.verbose and (generation % 5 == 0 or generation == self.generations - 1):
                print(f"Generacion {generation:03d} | fitness {population[0].fitness:.6f}")

            survivors = population[: self.elite_count]
            children: List[Individual] = []
            while len(survivors) + len(children) < self.population_size:
                parent_a = self.random.choice(survivors)
                if self.random.random() < self.crossover_rate:
                    parent_b = self.random.choice(survivors)
                    parent = self._crossover(parent_a, parent_b)
                else:
                    parent = parent_a
                sigma = self._annealed_sigma(generation)
                child = self._mutate(parent, sigma)
                children.append(self._guided_mutate(child, ratings))

            population = [self._clone(parent) for parent in survivors] + children

        for individual in population:
            individual.fitness = self._fitness(individual, ratings)
        population.sort(key=lambda candidate: candidate.fitness)
        self.best_ = self._clone(population[0])
        self._record_generation(population, self.generations)
        return self

    def predict(self, user_idx: int, item_idx: int) -> float:
        if self.best_ is None:
            raise RuntimeError("fit must be called before predict")
        return self._predict_with(self.best_, user_idx, item_idx)

    def recommend(
        self,
        user_idx: int,
        known_items: Iterable[int],
        top_n: int = 5,
    ) -> List[Tuple[int, float]]:
        """Top-N de ítems no conocidos, ordenados por preferencia estimada."""
        recommendations, _ = self.recommend_with_cutoff(user_idx, known_items, top_n)
        return recommendations

    def recommend_with_cutoff(
        self,
        user_idx: int,
        known_items: Iterable[int],
        top_n: int = 5,
    ) -> Tuple[List[Tuple[int, float]], float | None]:
        """Top-N y score crudo del mejor ítem no recomendado."""
        if self.best_ is None:
            raise RuntimeError("fit must be called before recommend")

        known = set(known_items)
        candidates = []
        for item_idx in range(self.n_items):
            if item_idx not in known:
                raw = self._raw_predict_with(self.best_, user_idx, item_idx)
                candidates.append((item_idx, raw))
        candidates.sort(key=lambda pair: pair[1], reverse=True)

        recommendations = [
            (item_idx, min(self.rating_max, max(self.rating_min, raw)))
            for item_idx, raw in candidates[:top_n]
        ]
        cutoff = candidates[top_n][1] if len(candidates) > top_n else None
        return recommendations, cutoff

    def prediction_components(self, user_idx: int, item_idx: int) -> Dict[str, Any]:
        if self.best_ is None:
            raise RuntimeError("fit must be called before prediction_components")

        factor_values = [
            user_value * item_value
            for user_value, item_value in zip(
                self.best_.user_factors[user_idx],
                self.best_.item_factors[item_idx],
            )
        ]
        latent_dot = sum(factor_values)
        raw_prediction = (
            self.best_.global_mean
            + self.best_.user_bias[user_idx]
            + self.best_.item_bias[item_idx]
            + latent_dot
        )
        return {
            "global_mean": self.best_.global_mean,
            "user_bias": self.best_.user_bias[user_idx],
            "item_bias": self.best_.item_bias[item_idx],
            "latent_dot": latent_dot,
            "raw_prediction": raw_prediction,
            "prediction": min(self.rating_max, max(self.rating_min, raw_prediction)),
            "factor_contributions": factor_values,
        }

    def _record_generation(
        self,
        population: Sequence[Individual],
        generation: int,
    ) -> None:
        fitness_values = [individual.fitness for individual in population]
        mean = sum(fitness_values) / len(fitness_values)
        variance = sum((value - mean) ** 2 for value in fitness_values) / len(fitness_values)
        stat = {
            "generation": float(generation),
            "best": fitness_values[0],
            "mean": mean,
            "worst": fitness_values[-1],
            "fitness_std": math.sqrt(variance),
            "mutation_sigma": self._annealed_sigma(min(generation, self.generations - 1)),
        }
        self.history_.append(stat["best"])
        self.generation_stats_.append(stat)

    def _baseline_terms(
        self,
        ratings: Sequence[Rating],
        shrinkage: float = 10.0,
    ) -> Tuple[float, List[float], List[float]]:
        global_mean = sum(rating for _, _, rating in ratings) / len(ratings)
        user_sum = [0.0 for _ in range(self.n_users)]
        user_count = [0 for _ in range(self.n_users)]
        item_sum = [0.0 for _ in range(self.n_items)]
        item_count = [0 for _ in range(self.n_items)]

        for user_idx, item_idx, rating in ratings:
            residual = rating - global_mean
            user_sum[user_idx] += residual
            user_count[user_idx] += 1
            item_sum[item_idx] += residual
            item_count[item_idx] += 1

        user_bias = [
            total / (count + shrinkage) if count else 0.0
            for total, count in zip(user_sum, user_count)
        ]
        item_bias = [
            total / (count + shrinkage) if count else 0.0
            for total, count in zip(item_sum, item_count)
        ]
        return global_mean, user_bias, item_bias

    def _guided_mutate(self, child: Individual, ratings: Sequence[Rating]) -> Individual:
        """Mutación dirigida por el residuo sobre un mini-lote."""
        if self.guidance_rate <= 0 or not ratings:
            return child

        batch = min(self.guidance_batch, len(ratings))
        n_ratings = len(ratings)
        rate = self.guidance_rate
        reg = self.regularization
        randrange = self.random.randrange

        for _ in range(batch):
            user_idx, item_idx, rating = ratings[randrange(n_ratings)]
            error = rating - self._raw_predict_with(child, user_idx, item_idx)
            if error == 0.0:
                continue

            user_vector = child.user_factors[user_idx]
            item_vector = child.item_factors[item_idx]
            for position in range(self.factors):
                user_value = user_vector[position]
                item_value = item_vector[position]
                user_vector[position] = user_value + rate * (
                    error * item_value - reg * user_value
                )
                item_vector[position] = item_value + rate * (
                    error * user_value - reg * item_value
                )

            bias_rate = rate * self.bias_guidance_ratio
            child.user_bias[user_idx] += bias_rate * (
                error - reg * child.user_bias[user_idx]
            )
            child.item_bias[item_idx] += bias_rate * (
                error - reg * child.item_bias[item_idx]
            )

        child.fitness = float("inf")
        return child

    def _new_individual(
        self,
        global_mean: float,
        user_bias_seed: Sequence[float],
        item_bias_seed: Sequence[float],
    ) -> Individual:
        scale = self.init_scale
        return Individual(
            user_factors=[
                [self.random.gauss(0.0, scale) for _ in range(self.factors)]
                for _ in range(self.n_users)
            ],
            item_factors=[
                [self.random.gauss(0.0, scale) for _ in range(self.factors)]
                for _ in range(self.n_items)
            ],
            user_bias=[
                value + self.random.gauss(0.0, 0.03)
                for value in user_bias_seed
            ],
            item_bias=[
                value + self.random.gauss(0.0, 0.03)
                for value in item_bias_seed
            ],
            global_mean=global_mean,
        )

    def _fitness(self, individual: Individual, ratings: Sequence[Rating]) -> float:
        squared_error = 0.0
        for user_idx, item_idx, rating in ratings:
            error = rating - self._predict_with(individual, user_idx, item_idx)
            squared_error += error * error

        regularization = self._regularization_penalty(individual)
        score = squared_error / len(ratings) + self.regularization * regularization

        if self.explainability_weight > 0.0:
            score -= self.explainability_weight * self._explainability_term(individual)
        return score

    def _explainability_term(self, individual: Individual) -> float:
        """Respaldo medio en evidencia de las recomendaciones Top-N."""
        if not self._explain_users:
            return 0.0

        total = 0.0
        cuenta = 0
        for user_idx in self._explain_users:
            conocidos = self._known_by_user.get(user_idx, {})
            if not conocidos:
                continue

            candidatos = []
            for item_idx in range(self.n_items):
                if item_idx not in conocidos:
                    candidatos.append(
                        (self._raw_predict_with(individual, user_idx, item_idx), item_idx)
                    )
            candidatos.sort(reverse=True)

            for _, item_idx in candidatos[: self.explainability_top_n]:
                objetivo = individual.item_factors[item_idx]
                soporte = []
                for conocido_idx, valoracion in conocidos.items():
                    if valoracion < DEFAULT_MIN_EVIDENCE_RATING:
                        continue
                    similitud = cosine_similarity(
                        objetivo, individual.item_factors[conocido_idx]
                    )
                    if similitud > DEFAULT_MIN_EVIDENCE_SIMILARITY:
                        soporte.append(similitud * (valoracion - 1.0) / 4.0)
                soporte.sort(reverse=True)
                mejores = soporte[:5]
                total += sum(mejores) / len(mejores) if mejores else 0.0
                cuenta += 1

        return total / cuenta if cuenta else 0.0

    def _regularization_penalty(self, individual: Individual) -> float:
        total = 0.0
        for matrix in (individual.user_factors, individual.item_factors):
            for row in matrix:
                total += sum(value * value for value in row)
        total += sum(value * value for value in individual.user_bias)
        total += sum(value * value for value in individual.item_bias)
        return total

    def _raw_predict_with(
        self,
        individual: Individual,
        user_idx: int,
        item_idx: int,
    ) -> float:
        dot = sum(
            user_value * item_value
            for user_value, item_value in zip(
                individual.user_factors[user_idx],
                individual.item_factors[item_idx],
            )
        )
        return (
            individual.global_mean
            + individual.user_bias[user_idx]
            + individual.item_bias[item_idx]
            + dot
        )

    def _predict_with(self, individual: Individual, user_idx: int, item_idx: int) -> float:
        prediction = self._raw_predict_with(individual, user_idx, item_idx)
        return min(self.rating_max, max(self.rating_min, prediction))

    def _annealed_sigma(self, generation: int) -> float:
        progress = generation / max(1, self.generations - 1)
        return self.mutation_sigma * (1.0 - 0.75 * progress)

    def _mutate(self, parent: Individual, sigma: float) -> Individual:
        child = self._clone(parent)
        child.user_factors = self._mutate_matrix(child.user_factors, sigma)
        child.item_factors = self._mutate_matrix(child.item_factors, sigma)
        child.user_bias = self._mutate_vector(child.user_bias, sigma * 0.35)
        child.item_bias = self._mutate_vector(child.item_bias, sigma * 0.35)
        child.fitness = float("inf")
        return child

    def _mutate_matrix(self, matrix: List[List[float]], sigma: float) -> List[List[float]]:
        return [
            [
                value + self.random.gauss(0.0, sigma)
                if self.random.random() < self.mutation_rate
                else value
                for value in row
            ]
            for row in matrix
        ]

    def _mutate_vector(self, values: List[float], sigma: float) -> List[float]:
        return [
            value + self.random.gauss(0.0, sigma)
            if self.random.random() < self.mutation_rate
            else value
            for value in values
        ]

    def _crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        return Individual(
            user_factors=[
                row_a[:] if self.random.random() < 0.5 else row_b[:]
                for row_a, row_b in zip(parent_a.user_factors, parent_b.user_factors)
            ],
            item_factors=[
                row_a[:] if self.random.random() < 0.5 else row_b[:]
                for row_a, row_b in zip(parent_a.item_factors, parent_b.item_factors)
            ],
            user_bias=[
                value_a if self.random.random() < 0.5 else value_b
                for value_a, value_b in zip(parent_a.user_bias, parent_b.user_bias)
            ],
            item_bias=[
                value_a if self.random.random() < 0.5 else value_b
                for value_a, value_b in zip(parent_a.item_bias, parent_b.item_bias)
            ],
            global_mean=(parent_a.global_mean + parent_b.global_mean) / 2.0,
        )

    def _clone(self, individual: Individual) -> Individual:
        return Individual(
            user_factors=[row[:] for row in individual.user_factors],
            item_factors=[row[:] for row in individual.item_factors],
            user_bias=individual.user_bias[:],
            item_bias=individual.item_bias[:],
            global_mean=individual.global_mean,
            fitness=individual.fitness,
        )


# CARGA DE DATOS
def load_ratings_csv(path: Path) -> Dataset:
    """Carga un CSV con columnas user_id, item_id, rating."""

    user_to_idx: Dict[str, int] = {}
    item_to_idx: Dict[str, int] = {}
    idx_to_user: List[str] = []
    idx_to_item: List[str] = []
    ratings: List[Rating] = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"user_id", "item_id", "rating"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("CSV must contain user_id,item_id,rating columns")

        for row in reader:
            user_id = row["user_id"]
            item_id = row["item_id"]
            rating = float(row["rating"])

            if user_id not in user_to_idx:
                user_to_idx[user_id] = len(idx_to_user)
                idx_to_user.append(user_id)
            if item_id not in item_to_idx:
                item_to_idx[item_id] = len(idx_to_item)
                idx_to_item.append(item_id)

            ratings.append((user_to_idx[user_id], item_to_idx[item_id], rating))

    return Dataset(ratings, user_to_idx, item_to_idx, idx_to_user, idx_to_item)


def load_interactions_csv(
    path: Path,
    user_column: str,
    item_column: str,
    rating_column: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    drop_zero_ratings: bool = False,
    rating_transform: str = "none",
    item_titles_by_external_id: Dict[str, str] | None = None,
) -> Dataset:
    """Carga un CSV de interacciones con columnas configurables."""

    raw_rows: List[Tuple[str, str, float]] = []
    with path.open(newline="", encoding=encoding) as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        required = {user_column, item_column, rating_column}
        if not required.issubset(reader.fieldnames or set()):
            columns = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"CSV must contain {required}. Found columns: {columns}"
            )

        for row in reader:
            rating = float(row[rating_column])
            if drop_zero_ratings and rating == 0:
                continue
            raw_rows.append((row[user_column], row[item_column], rating))

    if not raw_rows:
        raise ValueError("CSV did not contain usable interactions")

    transformed_rows = transform_ratings(raw_rows, rating_transform)
    return build_dataset(transformed_rows, item_titles_by_external_id)


def load_lastfm_hetrec_folder(
    path: Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> Dataset:
    """Carga Last.fm HetRec2011 (user_artists.dat y artists.dat)."""
    ratings_path = path / "user_artists.dat"
    artists_path = path / "artists.dat"
    if not ratings_path.exists():
        raise FileNotFoundError(f"HetRec Last.fm file not found: {ratings_path}")

    item_titles = None
    if artists_path.exists():
        item_titles = load_item_titles_csv(
            artists_path,
            item_id_column="id",
            item_name_column="name",
            delimiter="\t",
            encoding="utf-8",
        )

    dataset = load_interactions_csv(
        ratings_path,
        user_column="userID",
        item_column="artistID",
        rating_column="weight",
        delimiter="\t",
        encoding="utf-8",
        drop_zero_ratings=False,
        rating_transform="log-scale-1-5",
        item_titles_by_external_id=item_titles,
    )
    return filter_dataset(
        dataset,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )


def load_bookcrossing_folder(
    path: Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> Dataset:
    ratings_path = path / "Ratings.csv"
    books_path = path / "Books.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Book-Crossing ratings file not found: {ratings_path}")

    item_titles = None
    if books_path.exists():
        item_titles = load_item_titles_csv(
            books_path,
            item_id_column="ISBN",
            item_name_column="Title",
            delimiter=";",
            encoding="utf-8",
        )

    dataset = load_interactions_csv(
        ratings_path,
        user_column="User-ID",
        item_column="ISBN",
        rating_column="Rating",
        delimiter=";",
        encoding="utf-8",
        drop_zero_ratings=True,
        rating_transform="scale-1-5",
        item_titles_by_external_id=item_titles,
    )
    return filter_dataset(
        dataset,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )


def filter_dataset(
    dataset: Dataset,
    min_user_interactions: int = 1,
    min_item_interactions: int = 1,
) -> Dataset:
    if min_user_interactions <= 1 and min_item_interactions <= 1:
        return dataset

    ratings = dataset.ratings
    while True:
        user_counts: Dict[int, int] = {}
        item_counts: Dict[int, int] = {}
        for user_idx, item_idx, _ in ratings:
            user_counts[user_idx] = user_counts.get(user_idx, 0) + 1
            item_counts[item_idx] = item_counts.get(item_idx, 0) + 1

        filtered = [
            (user_idx, item_idx, rating)
            for user_idx, item_idx, rating in ratings
            if user_counts.get(user_idx, 0) >= min_user_interactions
            and item_counts.get(item_idx, 0) >= min_item_interactions
        ]
        if len(filtered) == len(ratings):
            break
        ratings = filtered

    rows = [
        (
            dataset.idx_to_user[user_idx],
            dataset.idx_to_item[item_idx],
            rating,
        )
        for user_idx, item_idx, rating in ratings
    ]
    item_titles_by_external_id = None
    if dataset.item_titles:
        item_titles_by_external_id = {
            dataset.idx_to_item[item_idx]: title
            for item_idx, title in dataset.item_titles.items()
        }
    return build_dataset(rows, item_titles_by_external_id)


def transform_ratings(
    rows: Sequence[Tuple[str, str, float]],
    transform: str,
) -> List[Tuple[str, str, float]]:
    if transform == "none":
        return list(rows)

    if transform == "scale-1-5":
        values = [rating for _, _, rating in rows]
        low, high = min(values), max(values)
        if low == high:
            return [(user_id, item_id, 3.0) for user_id, item_id, _ in rows]
        return [
            (user_id, item_id, 1.0 + 4.0 * (rating - low) / (high - low))
            for user_id, item_id, rating in rows
        ]

    if transform == "log-scale-1-5":
        logged = [
            (user_id, item_id, math.log1p(max(0.0, rating)))
            for user_id, item_id, rating in rows
        ]
        return transform_ratings(logged, "scale-1-5")

    raise ValueError(
        "rating_transform must be one of: none, scale-1-5, log-scale-1-5"
    )


def build_dataset(
    rows: Sequence[Tuple[str, str, float]],
    item_titles_by_external_id: Dict[str, str] | None = None,
) -> Dataset:
    user_to_idx: Dict[str, int] = {}
    item_to_idx: Dict[str, int] = {}
    idx_to_user: List[str] = []
    idx_to_item: List[str] = []
    ratings: List[Rating] = []

    for user_id, item_id, rating in rows:
        if user_id not in user_to_idx:
            user_to_idx[user_id] = len(idx_to_user)
            idx_to_user.append(user_id)
        if item_id not in item_to_idx:
            item_to_idx[item_id] = len(idx_to_item)
            idx_to_item.append(item_id)
        ratings.append((user_to_idx[user_id], item_to_idx[item_id], rating))

    item_titles = None
    if item_titles_by_external_id:
        item_titles = {
            item_idx: item_titles_by_external_id[item_id]
            for item_id, item_idx in item_to_idx.items()
            if item_id in item_titles_by_external_id
        }

    return Dataset(ratings, user_to_idx, item_to_idx, idx_to_user, idx_to_item, item_titles)


def load_item_titles_csv(
    path: Path,
    item_id_column: str,
    item_name_column: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> Dict[str, str]:
    with path.open(newline="", encoding=encoding) as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        required = {item_id_column, item_name_column}
        if not required.issubset(reader.fieldnames or set()):
            columns = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"Items CSV must contain {required}. Found columns: {columns}"
            )
        return {
            row[item_id_column]: row[item_name_column]
            for row in reader
            if row.get(item_id_column) and row.get(item_name_column)
        }


def load_movielens_100k(path: Path) -> Dataset:
    """Carga MovieLens 100K (u.data y, si existe, u.item)."""

    data_path = path / "u.data"
    item_path = path / "u.item"
    if not data_path.exists():
        raise FileNotFoundError(f"MovieLens file not found: {data_path}")

    user_to_idx: Dict[str, int] = {}
    item_to_idx: Dict[str, int] = {}
    idx_to_user: List[str] = []
    idx_to_item: List[str] = []
    ratings: List[Rating] = []

    with data_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                raise ValueError(f"Invalid u.data row at line {line_number}")

            user_id, item_id, rating_text = parts[:3]
            if user_id not in user_to_idx:
                user_to_idx[user_id] = len(idx_to_user)
                idx_to_user.append(user_id)
            if item_id not in item_to_idx:
                item_to_idx[item_id] = len(idx_to_item)
                idx_to_item.append(item_id)

            ratings.append(
                (user_to_idx[user_id], item_to_idx[item_id], float(rating_text))
            )

    item_titles = load_movielens_titles(item_path, item_to_idx) if item_path.exists() else {}
    return Dataset(ratings, user_to_idx, item_to_idx, idx_to_user, idx_to_item, item_titles)


def load_movielens_titles(path: Path, item_to_idx: Dict[str, int]) -> Dict[int, str]:
    titles: Dict[int, str] = {}
    with path.open(encoding="latin-1") as file:
        for line in file:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2 and parts[0] in item_to_idx:
                titles[item_to_idx[parts[0]]] = parts[1]
    return titles


def demo_dataset() -> Dataset:
    rows = [
        ("ana", "matrix", 5), ("ana", "arrival", 4), ("ana", "coco", 1),
        ("ana", "amelie", 2), ("ana", "parasite", 5),
        ("bruno", "matrix", 4), ("bruno", "arrival", 5), ("bruno", "coco", 1),
        ("bruno", "parasite", 4), ("bruno", "blade_runner", 5),
        ("carla", "coco", 5), ("carla", "amelie", 4), ("carla", "totoro", 5),
        ("carla", "matrix", 1), ("carla", "arrival", 2),
        ("diego", "matrix", 5), ("diego", "blade_runner", 5), ("diego", "arrival", 4),
        ("diego", "totoro", 1), ("diego", "parasite", 4),
        ("elena", "coco", 4), ("elena", "amelie", 5), ("elena", "totoro", 5),
        ("elena", "parasite", 2), ("elena", "arrival", 2),
        ("fabio", "parasite", 5), ("fabio", "matrix", 4), ("fabio", "blade_runner", 4),
        ("fabio", "amelie", 2), ("fabio", "arrival", 4),
    ]

    user_to_idx: Dict[str, int] = {}
    item_to_idx: Dict[str, int] = {}
    idx_to_user: List[str] = []
    idx_to_item: List[str] = []
    ratings: List[Rating] = []

    for user_id, item_id, rating in rows:
        if user_id not in user_to_idx:
            user_to_idx[user_id] = len(idx_to_user)
            idx_to_user.append(user_id)
        if item_id not in item_to_idx:
            item_to_idx[item_id] = len(idx_to_item)
            idx_to_item.append(item_id)
        ratings.append((user_to_idx[user_id], item_to_idx[item_id], float(rating)))

    return Dataset(ratings, user_to_idx, item_to_idx, idx_to_user, idx_to_item)


# UTILIDADES
def known_items_by_user(ratings: Sequence[Rating]) -> Dict[int, set[int]]:
    known: Dict[int, set[int]] = {}
    for user_idx, item_idx, _ in ratings:
        known.setdefault(user_idx, set()).add(item_idx)
    return known


def rmse(model: EvolutionaryCollaborativeFilter, ratings: Sequence[Rating]) -> float:
    mse = sum(
        (rating - model.predict(user_idx, item_idx)) ** 2
        for user_idx, item_idx, rating in ratings
    ) / len(ratings)
    return math.sqrt(mse)


def item_label(dataset: Dataset, item_idx: int) -> str:
    if dataset.item_titles:
        return dataset.item_titles.get(item_idx, dataset.idx_to_item[item_idx])
    return dataset.idx_to_item[item_idx]


def ratings_for_user(ratings: Sequence[Rating], user_idx: int) -> List[Tuple[int, float]]:
    return [
        (item_idx, rating)
        for current_user_idx, item_idx, rating in ratings
        if current_user_idx == user_idx
    ]


# GENERACIÓN DE VISUALIZACIONES
def generate_explanation_plots(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_id: str,
    recommendations: Sequence[Tuple[int, float]],
    output_dir: Path,
) -> List[Path]:
    """Genera las visualizaciones de la capa técnica."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate explanation plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    user_idx = dataset.user_to_idx[user_id]
    created = [
        plot_training_dynamics(model, output_dir / "01_training_dynamics.png", plt),
        plot_recommendation_scores(
            dataset,
            recommendations,
            output_dir / "02_recommendation_scores.png",
            plt,
        ),
        plot_score_breakdown(
            model,
            dataset,
            user_idx,
            recommendations,
            output_dir / "03_score_breakdown.png",
            plt,
        ),
        plot_user_profile_vs_recommendations(
            model,
            dataset,
            user_idx,
            recommendations,
            output_dir / "04_user_profile_vs_recommendations.png",
            plt,
        ),
        plot_latent_space(
            model,
            dataset,
            user_idx,
            recommendations,
            output_dir / "05_latent_space.png",
            plt,
        ),
    ]
    return created


def generate_user_explanation_plots(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_id: str,
    recommendations: Sequence[Tuple[int, float]],
    output_dir: Path,
) -> List[Path]:
    """Genera las visualizaciones de la capa de usuario final."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate user explanation plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    user_idx = dataset.user_to_idx[user_id]
    known = ratings_for_user(dataset.ratings, user_idx)
    evidence = build_recommendation_evidence(model, dataset, user_idx, recommendations)
    created = [
        plot_user_why_recommended(
            dataset,
            evidence,
            output_dir / "01_why_recommended.png",
            plt,
        ),
        plot_similar_items_evidence(
            dataset,
            evidence,
            output_dir / "02_similar_items_evidence.png",
            plt,
        ),
        plot_user_profile_match(
            dataset,
            known,
            recommendations,
            output_dir / "03_user_profile_match.png",
            plt,
        ),
        plot_recommendation_confidence(
            dataset,
            evidence,
            output_dir / "04_recommendation_confidence.png",
            plt,
        ),
        plot_user_friendly_latent_map(
            model,
            dataset,
            user_idx,
            recommendations,
            output_dir / "05_user_friendly_map.png",
            plt,
        ),
        plot_latent_factor_contributions(
            dataset,
            evidence,
            output_dir / "06_latent_factor_contributions.png",
            plt,
        ),
        plot_user_item_factor_radar(
            model,
            dataset,
            user_idx,
            recommendations,
            output_dir / "07_user_item_factor_radar.png",
            plt,
        ),
    ]
    return created


# EVIDENCIA Y CONFIANZA
def build_recommendation_evidence(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    recommendations: Sequence[Tuple[int, float]],
    min_similarity: float = DEFAULT_MIN_EVIDENCE_SIMILARITY,
    min_rating: float = DEFAULT_MIN_EVIDENCE_RATING,
) -> List[Dict[str, Any]]:
    known = ratings_for_user(dataset.ratings, user_idx)
    known_by_item = {item_idx: rating for item_idx, rating in known}
    item_popularity = item_rating_counts(dataset.ratings)
    user_activity = len(known)
    max_popularity = max(item_popularity.values()) if item_popularity else 1
    evidence = []

    for item_idx, score in recommendations:
        similar_known = most_similar_known_items(
            model,
            dataset,
            item_idx,
            known_by_item,
            top_n=5,
            min_similarity=min_similarity,
            min_rating=min_rating,
        )
        preference_match = weighted_similarity_score(similar_known)
        popularity_score = item_popularity.get(item_idx, 0) / max_popularity
        signals = confidence_signals(
            user_activity=user_activity,
            item_count=item_popularity.get(item_idx, 0),
            max_item_count=max_popularity,
            preference_match=preference_match,
        )
        confidence = sum(
            w * s for w, s in zip(DEFAULT_CONFIDENCE_WEIGHTS, signals)
        )
        components = model.prediction_components(user_idx, item_idx)
        evidence.append(
            {
                "item_idx": item_idx,
                "score": score,
                "similar_known": similar_known,
                "preference_match": preference_match,
                "popularity_score": popularity_score,
                "confidence": confidence,
                "confidence_signals": signals,
                "components": components,
            }
        )
    return evidence


def item_rating_counts(ratings: Sequence[Rating]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for _, item_idx, _ in ratings:
        counts[item_idx] = counts.get(item_idx, 0) + 1
    return counts


def most_similar_known_items(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    recommended_item_idx: int,
    known_by_item: Dict[int, float],
    top_n: int = 5,
    min_similarity: float = DEFAULT_MIN_EVIDENCE_SIMILARITY,
    min_rating: float = DEFAULT_MIN_EVIDENCE_RATING,
) -> List[Tuple[int, float, float]]:
    """Ítems del historial que respaldan la recomendación."""
    if model.best_ is None:
        raise RuntimeError("fit must be called before most_similar_known_items")

    target = model.best_.item_factors[recommended_item_idx]
    similarities = []
    for item_idx, rating in known_by_item.items():
        if rating < min_rating:
            continue
        candidate = model.best_.item_factors[item_idx]
        similarity = cosine_similarity(target, candidate)
        if similarity <= min_similarity:
            continue
        similarities.append((item_idx, similarity, rating))
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities[:top_n]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def weighted_similarity_score(similar_known: Sequence[Tuple[int, float, float]]) -> float:
    if not similar_known:
        return 0.0
    weighted = [
        max(0.0, similarity) * max(0.0, (rating - 1.0) / 4.0)
        for _, similarity, rating in similar_known
    ]
    return min(1.0, sum(weighted) / max(1, len(weighted)))


def confidence_signals(
    user_activity: int,
    item_count: int,
    max_item_count: int,
    preference_match: float,
) -> Tuple[float, float, float]:
    """Las tres señales de la confianza, sin ponderar."""
    user_evidence = min(
        1.0, math.log1p(user_activity) / math.log1p(CONFIDENCE_ACTIVITY_REFERENCE)
    )
    item_evidence = min(
        1.0, math.log1p(item_count) / math.log1p(max(2, max_item_count))
    )
    return preference_match, user_evidence, item_evidence


def recommendation_confidence(
    user_activity: int,
    item_count: int,
    max_item_count: int,
    preference_match: float,
    weights: Sequence[float] = DEFAULT_CONFIDENCE_WEIGHTS,
) -> float:
    """Confianza explicativa: combinación ponderada de tres señales."""
    senales = confidence_signals(
        user_activity, item_count, max_item_count, preference_match
    )
    return sum(w * s for w, s in zip(weights, senales))


# VISUALIZACIONES PARA EL USUARIO FINAL
def plot_user_why_recommended(
    dataset: Dataset,
    evidence: Sequence[Dict[str, Any]],
    path: Path,
    plt: Any,
) -> Path:
    labels = [short_label(item_label(dataset, entry["item_idx"])) for entry in evidence]
    preference = [entry["preference_match"] for entry in evidence]
    popularity = [entry["popularity_score"] for entry in evidence]
    predicted = [(entry["score"] - 1.0) / 4.0 for entry in evidence]

    x_positions = list(range(len(labels)))
    width = 0.25
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar([x - width for x in x_positions], preference, width, label="Se parece a tus gustos")
    axis.bar(x_positions, predicted, width, label="Prediccion personalizada")
    axis.bar([x + width for x in x_positions], popularity, width, label="Evidencia en la comunidad")
    axis.set_title("Por que aparecen estas recomendaciones")
    axis.set_ylabel("Fuerza de la evidencia")
    axis.set_ylim(0, 1)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_similar_items_evidence(
    dataset: Dataset,
    evidence: Sequence[Dict[str, Any]],
    path: Path,
    plt: Any,
) -> Path:
    rows = []
    for entry in evidence[:5]:
        recommended = short_label(item_label(dataset, entry["item_idx"]), 24)
        for item_idx, similarity, rating in entry["similar_known"][:3]:
            rows.append(
                (
                    f"{recommended}\npor {short_label(item_label(dataset, item_idx), 24)}",
                    max(0.0, similarity),
                    rating,
                )
            )

    fig, axis = plt.subplots(figsize=(10, max(4, 0.45 * len(rows))))
    if rows:
        labels = [row[0] for row in rows]
        similarities = [row[1] for row in rows]
        colors = [rating_color(row[2]) for row in rows]
        axis.barh(labels[::-1], similarities[::-1], color=colors[::-1])
        axis.set_xlim(0, 1)
    axis.set_title("Se parece a cosas que ya valoraste")
    axis.set_xlabel("Similitud latente")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_user_profile_match(
    dataset: Dataset,
    known: Sequence[Tuple[int, float]],
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    liked = sorted(known, key=lambda pair: pair[1], reverse=True)[:8]
    disliked = sorted(known, key=lambda pair: pair[1])[:8]
    recs = list(recommendations[:8])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    profile_panels = [
        ("Lo que mas te gusto", liked, "#59a14f", "Rating observado"),
        ("Lo que menos encajo", disliked, "#e15759", "Rating observado"),
        ("Nuevas sugerencias", recs, "#4c78a8", "Score predicho"),
    ]
    for axis, (title, rows, color, xlabel) in zip(axes, profile_panels):
        labels = [short_label(item_label(dataset, item_idx), 24) for item_idx, _ in rows]
        values = [value for _, value in rows]
        axis.barh(labels[::-1], values[::-1], color=color)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_xlim(1, 5)
        axis.grid(axis="x", alpha=0.25)

    fig.suptitle("Tu perfil y las recomendaciones", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recommendation_confidence(
    dataset: Dataset,
    evidence: Sequence[Dict[str, Any]],
    path: Path,
    plt: Any,
) -> Path:
    labels = [short_label(item_label(dataset, entry["item_idx"])) for entry in evidence]
    values = [entry["confidence"] for entry in evidence]
    colors = [confidence_color(value) for value in values]

    fig, axis = plt.subplots(figsize=(9, max(4, 0.45 * len(values))))
    axis.barh(labels[::-1], values[::-1], color=colors[::-1])
    axis.set_title("Que tan fuerte es la evidencia")
    axis.set_xlabel("Confianza explicativa")
    axis.set_xlim(0, 1)
    for umbral in DEFAULT_CONFIDENCE_THRESHOLDS:
        axis.axvline(umbral, color="#999999", linestyle="--", linewidth=0.8)
    axis.text(0.16, -0.7, "Exploratoria", ha="center", fontsize=9)
    axis.text(0.50, -0.7, "Media", ha="center", fontsize=9)
    axis.text(0.83, -0.7, "Alta", ha="center", fontsize=9)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_user_friendly_latent_map(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    if model.best_ is None:
        raise RuntimeError("fit must be called before plot_user_friendly_latent_map")

    known = sorted(ratings_for_user(dataset.ratings, user_idx), key=lambda pair: pair[1], reverse=True)[:20]
    recommended_items = [item_idx for item_idx, _ in recommendations[:8]]
    user_x, user_y = two_factor_point(model.best_.user_factors[user_idx])

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter([user_x], [user_y], marker="*", s=260, color="#e45756", label="Tu perfil")

    if known:
        known_points = [two_factor_point(model.best_.item_factors[item_idx]) for item_idx, _ in known]
        axis.scatter(
            [point[0] for point in known_points],
            [point[1] for point in known_points],
            s=45,
            color="#59a14f",
            alpha=0.70,
            label="Items que ya conoces",
        )

    if recommended_items:
        rec_points = [two_factor_point(model.best_.item_factors[item_idx]) for item_idx in recommended_items]
        axis.scatter(
            [point[0] for point in rec_points],
            [point[1] for point in rec_points],
            s=80,
            color="#4c78a8",
            label="Recomendaciones",
        )
        for item_idx, (x_value, y_value) in zip(recommended_items, rec_points):
            axis.annotate(short_label(item_label(dataset, item_idx), 18), (x_value, y_value), fontsize=8)

    axis.set_title("Donde caen tus recomendaciones frente a tus gustos")
    axis.set_xlabel("Dimension de gusto 1")
    axis.set_ylabel("Dimension de gusto 2")
    axis.grid(True, alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_latent_factor_contributions(
    dataset: Dataset,
    evidence: Sequence[Dict[str, Any]],
    path: Path,
    plt: Any,
) -> Path:
    if not evidence:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.set_title("Contribucion de factores latentes")
        axis.text(0.5, 0.5, "No hay recomendaciones para explicar", ha="center")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    selected = list(evidence[: min(5, len(evidence))])
    factors = selected[0]["components"]["factor_contributions"]
    factor_count = len(factors)
    labels = [f"F{i + 1}" for i in range(factor_count)]
    x_positions = list(range(factor_count))

    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(10, max(3, 2.1 * len(selected))),
        sharex=True,
    )
    if len(selected) == 1:
        axes = [axes]

    for axis, entry in zip(axes, selected):
        contributions = entry["components"]["factor_contributions"]
        colors = ["#59a14f" if value >= 0 else "#e15759" for value in contributions]
        axis.bar(x_positions, contributions, color=colors)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title(short_label(item_label(dataset, entry["item_idx"]), 60), fontsize=10)
        axis.set_ylabel("Aporte")
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("Factores latentes")
    fig.suptitle("Que factores latentes empujan o frenan cada recomendacion", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_user_item_factor_radar(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    if model.best_ is None:
        raise RuntimeError("fit must be called before plot_user_item_factor_radar")

    if not recommendations:
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.set_title("Radar de perfil latente")
        axis.text(0.5, 0.5, "No hay recomendaciones para explicar", ha="center")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path

    item_idx = recommendations[0][0]
    user_values = normalize_for_radar(model.best_.user_factors[user_idx])
    item_values = normalize_for_radar(model.best_.item_factors[item_idx])
    factor_count = len(user_values)
    angles = [2 * math.pi * idx / factor_count for idx in range(factor_count)]
    angles += angles[:1]
    user_plot = user_values + user_values[:1]
    item_plot = item_values + item_values[:1]
    ring = [2 * math.pi * idx / 200 for idx in range(201)]

    fig, axis = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    axis.plot(angles, user_plot, color="#4c78a8", linewidth=2, label="Tu perfil")
    axis.fill(angles, user_plot, color="#4c78a8", alpha=0.25)
    axis.plot(
        angles,
        item_plot,
        color="#f0a04b",
        linewidth=2,
        label=short_label(item_label(dataset, item_idx), 28),
    )
    axis.fill(angles, item_plot, color="#f0a04b", alpha=0.25)
    axis.plot(
        ring,
        [0.5] * len(ring),
        "--",
        color="#555555",
        linewidth=1.2,
        label="Cero (dentro: negativo)",
    )
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([f"F{i + 1}" for i in range(factor_count)])
    axis.set_ylim(0, 1)
    axis.set_yticks([0, 0.5, 1])
    axis.set_yticklabels(["-máx", "0", "+máx"], fontsize=8)
    axis.set_title("Perfil latente: usuario frente a ítem recomendado", pad=22)
    axis.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def normalize_for_radar(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    peak = max(abs(value) for value in values) or 1.0
    return [(value / peak + 1.0) / 2.0 for value in values]


def rating_color(rating: float) -> str:
    if rating >= 4:
        return "#59a14f"
    if rating >= 3:
        return "#f28e2b"
    return "#e15759"


def confidence_color(
    value: float,
    thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
) -> str:
    bajo, alto = thresholds
    if value >= alto:
        return "#59a14f"
    if value >= bajo:
        return "#f28e2b"
    return "#e15759"


# VISUALIZACIONES TÉCNICAS
def plot_training_dynamics(
    model: EvolutionaryCollaborativeFilter,
    path: Path,
    plt: Any,
) -> Path:
    generations = [stat["generation"] for stat in model.generation_stats_]
    best = [stat["best"] for stat in model.generation_stats_]
    mean = [stat["mean"] for stat in model.generation_stats_]
    worst = [stat["worst"] for stat in model.generation_stats_]
    sigma = [stat["mutation_sigma"] for stat in model.generation_stats_]

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(generations, best, label="Mejor fitness", linewidth=2.2)
    axis.plot(generations, mean, label="Fitness promedio", linewidth=1.6)
    axis.plot(generations, worst, label="Peor fitness", linewidth=1.0, alpha=0.65)
    axis.set_title("Evolucion del proceso evolutivo")
    axis.set_xlabel("Generacion")
    axis.set_ylabel("Fitness: error + regularizacion")
    axis.grid(True, alpha=0.25)

    sigma_axis = axis.twinx()
    sigma_axis.plot(generations, sigma, label="Sigma mutacion", linestyle="--", color="tab:red")
    sigma_axis.set_ylabel("Intensidad de mutacion")

    lines, labels = axis.get_legend_handles_labels()
    sigma_lines, sigma_labels = sigma_axis.get_legend_handles_labels()
    axis.legend(lines + sigma_lines, labels + sigma_labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_recommendation_scores(
    dataset: Dataset,
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    labels = [short_label(item_label(dataset, item_idx)) for item_idx, _ in recommendations]
    scores = [score for _, score in recommendations]

    fig, axis = plt.subplots(figsize=(9, max(4, 0.45 * len(scores))))
    axis.barh(labels[::-1], scores[::-1], color="#4c78a8")
    axis.set_title("Ranking recomendado")
    axis.set_xlabel("Score predicho")
    axis.set_xlim(1, 5)
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_score_breakdown(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    labels = [short_label(item_label(dataset, item_idx)) for item_idx, _ in recommendations]
    components = [
        model.prediction_components(user_idx, item_idx)
        for item_idx, _ in recommendations
    ]
    component_names = ["user_bias", "item_bias", "latent_dot"]
    display_names = ["Sesgo usuario", "Sesgo item", "Afinidad latente"]
    colors = ["#f58518", "#54a24b", "#b279a2"]

    x_positions = list(range(len(labels)))
    fig, axis = plt.subplots(figsize=(11, 5))
    global_mean = components[0]["global_mean"] if components else 0.0
    axis.axhline(global_mean, color="#72b7b2", linewidth=2,
                 label=f"Media global ({global_mean:.3f})")

    upper = [component["global_mean"] for component in components]
    lower = list(upper)
    for name, display_name, color in zip(component_names, display_names, colors):
        values = [component[name] for component in components]
        positive = [max(value, 0.0) for value in values]
        negative = [min(value, 0.0) for value in values]
        axis.bar(x_positions, positive, bottom=upper, label=display_name, color=color)
        if any(value < 0 for value in negative):
            axis.bar(x_positions, negative, bottom=lower, color=color,
                     hatch="//", edgecolor="#333333", linewidth=0.5)
        upper = [base + value for base, value in zip(upper, positive)]
        lower = [base + value for base, value in zip(lower, negative)]

    predictions = [component["prediction"] for component in components]
    axis.plot(x_positions, predictions, color="#222222", marker="o", label="Prediccion final")
    extremes = [global_mean] + upper + lower + predictions
    low, high = min(extremes), max(extremes)
    axis.set_ylim(low - 0.08 * (high - low), high + 0.06 * (high - low))
    axis.set_title("Descomposicion del score recomendado")
    axis.set_ylabel("Score")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_user_profile_vs_recommendations(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    known = ratings_for_user(dataset.ratings, user_idx)
    known_sorted = sorted(known, key=lambda pair: pair[1], reverse=True)[:10]

    labels = [short_label(item_label(dataset, item_idx)) for item_idx, _ in known_sorted]
    ratings = [rating for _, rating in known_sorted]
    rec_labels = [short_label(item_label(dataset, item_idx)) for item_idx, _ in recommendations]
    rec_scores = [score for _, score in recommendations]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * max(len(labels), len(rec_labels)))))
    axes[0].barh(labels[::-1], ratings[::-1], color="#59a14f")
    axes[0].set_title("Preferencias conocidas del usuario")
    axes[0].set_xlabel("Rating observado")
    axes[0].set_xlim(1, 5)
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(rec_labels[::-1], rec_scores[::-1], color="#4c78a8")
    axes[1].set_title("Items recomendados")
    axes[1].set_xlabel("Score predicho")
    axes[1].set_xlim(1, 5)
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_latent_space(
    model: EvolutionaryCollaborativeFilter,
    dataset: Dataset,
    user_idx: int,
    recommendations: Sequence[Tuple[int, float]],
    path: Path,
    plt: Any,
) -> Path:
    if model.best_ is None:
        raise RuntimeError("fit must be called before plot_latent_space")

    known_items = [item_idx for item_idx, _ in ratings_for_user(dataset.ratings, user_idx)]
    recommended_items = [item_idx for item_idx, _ in recommendations]
    known_items = known_items[:30]

    fig, axis = plt.subplots(figsize=(8, 6))
    user_x, user_y = two_factor_point(model.best_.user_factors[user_idx])
    axis.scatter([user_x], [user_y], marker="*", s=240, color="#e45756", label="Usuario")

    if known_items:
        known_points = [two_factor_point(model.best_.item_factors[item_idx]) for item_idx in known_items]
        axis.scatter(
            [point[0] for point in known_points],
            [point[1] for point in known_points],
            s=45,
            color="#59a14f",
            alpha=0.75,
            label="Items conocidos",
        )

    if recommended_items:
        rec_points = [two_factor_point(model.best_.item_factors[item_idx]) for item_idx in recommended_items]
        axis.scatter(
            [point[0] for point in rec_points],
            [point[1] for point in rec_points],
            s=70,
            color="#4c78a8",
            label="Recomendaciones",
        )
        for item_idx, (x_value, y_value) in zip(recommended_items, rec_points):
            axis.annotate(short_label(item_label(dataset, item_idx), 18), (x_value, y_value), fontsize=8)

    axis.set_title("Mapa latente usuario-item")
    axis.set_xlabel("Factor latente 1")
    axis.set_ylabel("Factor latente 2")
    axis.axhline(0, color="#888888", linewidth=0.7)
    axis.axvline(0, color="#888888", linewidth=0.7)
    axis.grid(True, alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def short_label(value: str, max_length: int = 32) -> str:
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def two_factor_point(values: Sequence[float]) -> Tuple[float, float]:
    x_value = values[0] if values else 0.0
    y_value = values[1] if len(values) > 1 else 0.0
    return x_value, y_value


# LÍNEA DE COMANDOS
def run(args: argparse.Namespace) -> None:
    if args.dataset in {"bookcrossing", "lastfm"} and not args.csv and not args.data_dir:
        raise ValueError(
            f"--csv or --data-dir is required when --dataset {args.dataset} is used"
        )

    item_titles = None
    if args.items_csv:
        if args.dataset == "bookcrossing":
            default_item_id_col = args.item_col or "ISBN"
            default_item_name_col = "Book-Title"
            default_items_delimiter = ";"
        elif args.dataset == "lastfm":
            default_item_id_col = args.item_col or "artistID"
            default_item_name_col = "name"
            default_items_delimiter = ","
        else:
            default_item_id_col = args.item_col or "item_id"
            default_item_name_col = "title"
            default_items_delimiter = ","

        item_titles = load_item_titles_csv(
            Path(args.items_csv),
            item_id_column=args.item_id_col or default_item_id_col,
            item_name_column=args.item_name_col or default_item_name_col,
            delimiter=args.items_delimiter or args.delimiter or default_items_delimiter,
            encoding=args.items_encoding or args.encoding,
        )

    if args.data_dir and args.dataset == "bookcrossing":
        dataset = load_bookcrossing_folder(
            Path(args.data_dir),
            min_user_interactions=args.min_user_interactions or 5,
            min_item_interactions=args.min_item_interactions or 5,
        )
    elif args.data_dir and args.dataset == "lastfm":
        dataset = load_lastfm_hetrec_folder(
            Path(args.data_dir),
            min_user_interactions=args.min_user_interactions or 5,
            min_item_interactions=args.min_item_interactions or 5,
        )
    elif args.dataset == "bookcrossing":
        dataset = load_interactions_csv(
            Path(args.csv),
            user_column=args.user_col or "User-ID",
            item_column=args.item_col or "ISBN",
            rating_column=args.rating_col or "Book-Rating",
            delimiter=args.delimiter or ";",
            encoding=args.encoding,
            drop_zero_ratings=True if args.drop_zero_ratings is None else args.drop_zero_ratings,
            rating_transform=args.rating_transform or "scale-1-5",
            item_titles_by_external_id=item_titles,
        )
    elif args.dataset == "lastfm":
        dataset = load_interactions_csv(
            Path(args.csv),
            user_column=args.user_col or "userID",
            item_column=args.item_col or "artistID",
            rating_column=args.rating_col or "weight",
            delimiter=args.delimiter or ",",
            encoding=args.encoding,
            drop_zero_ratings=False if args.drop_zero_ratings is None else args.drop_zero_ratings,
            rating_transform=args.rating_transform or "log-scale-1-5",
            item_titles_by_external_id=item_titles,
        )
    elif args.movielens_100k:
        dataset = load_movielens_100k(Path(args.movielens_100k))
    elif args.csv:
        dataset = load_interactions_csv(
            Path(args.csv),
            user_column=args.user_col or "user_id",
            item_column=args.item_col or "item_id",
            rating_column=args.rating_col or "rating",
            delimiter=args.delimiter or ",",
            encoding=args.encoding,
            drop_zero_ratings=bool(args.drop_zero_ratings),
            rating_transform=args.rating_transform or "none",
            item_titles_by_external_id=item_titles,
        )
        dataset = filter_dataset(
            dataset,
            min_user_interactions=args.min_user_interactions or 1,
            min_item_interactions=args.min_item_interactions or 1,
        )
    else:
        dataset = demo_dataset()
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
        seed=args.seed,
        verbose=args.verbose,
    )
    model.fit(dataset.ratings)

    print(f"Usuarios: {len(dataset.idx_to_user)}")
    print(f"Items: {len(dataset.idx_to_item)}")
    print(f"Ratings: {len(dataset.ratings)}")
    print(f"RMSE entrenamiento: {rmse(model, dataset.ratings):.4f}")
    print(f"Mejor fitness: {model.best_.fitness:.4f}")

    user_id = args.user or dataset.idx_to_user[0]
    if user_id not in dataset.user_to_idx:
        raise ValueError(f"Unknown user: {user_id}")

    user_idx = dataset.user_to_idx[user_id]
    known = known_items_by_user(dataset.ratings)
    recommendations = model.recommend(user_idx, known.get(user_idx, set()), args.top_n)

    print(f"\nRecomendaciones para {user_id}:")
    for item_idx, score in recommendations:
        item_name = item_label(dataset, item_idx)
        print(f"- {item_name}: {score:.2f}")

    if args.explain_plots:
        output_dir = Path(args.plots_dir)
        created = generate_explanation_plots(
            model,
            dataset,
            user_id,
            recommendations,
            output_dir,
        )
        print("\nGraficos de explicabilidad generados:")
        for path in created:
            print(f"- {path}")

    if args.user_explain_plots:
        output_dir = Path(args.user_plots_dir)
        created = generate_user_explanation_plots(
            model,
            dataset,
            user_id,
            recommendations,
            output_dir,
        )
        print("\nGraficos para usuario final generados:")
        for path in created:
            print(f"- {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de recomendacion colaborativo con programacion evolutiva."
    )
    parser.add_argument("--csv", help="CSV con columnas user_id,item_id,rating")
    parser.add_argument(
        "--data-dir",
        help="Carpeta del dataset: Ratings.csv y Books.csv (bookcrossing) o user_artists.dat (lastfm)",
    )
    parser.add_argument(
        "--dataset",
        choices=["generic", "bookcrossing", "lastfm"],
        default="generic",
        help="Preset de columnas y transformacion para datasets conocidos",
    )
    parser.add_argument("--user-col", help="Nombre de la columna de usuario")
    parser.add_argument("--item-col", help="Nombre de la columna de item")
    parser.add_argument("--rating-col", help="Nombre de la columna de rating/interaccion")
    parser.add_argument("--delimiter", help="Delimitador del CSV, por ejemplo ',' o ';'")
    parser.add_argument("--encoding", default="utf-8", help="Encoding del CSV")
    parser.add_argument("--items-csv", help="CSV opcional con nombres/titulos de items")
    parser.add_argument("--item-id-col", help="Columna id del item en --items-csv")
    parser.add_argument("--item-name-col", help="Columna nombre/titulo del item en --items-csv")
    parser.add_argument("--items-delimiter", help="Delimitador de --items-csv")
    parser.add_argument("--items-encoding", help="Encoding de --items-csv")
    parser.add_argument(
        "--drop-zero-ratings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Elimina interacciones con rating 0",
    )
    parser.add_argument(
        "--rating-transform",
        choices=["none", "scale-1-5", "log-scale-1-5"],
        help="Transformacion de ratings antes de entrenar",
    )
    parser.add_argument(
        "--guidance-rate",
        type=float,
        default=0.01,
        help="Intensidad del paso de mutacion dirigido por residuo. 0 desactiva la guia "
             "y recupera la mutacion puramente aleatoria",
    )
    parser.add_argument(
        "--guidance-batch",
        type=int,
        default=8192,
        help="Interacciones muestreadas por descendiente para el paso dirigido",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        help="Desviacion de la inicializacion de factores latentes (por defecto 0.3)",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        help="Minimo de interacciones por usuario despues de cargar el dataset",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        help="Minimo de interacciones por item despues de cargar el dataset",
    )
    parser.add_argument(
        "--movielens-100k",
        help="Carpeta descomprimida de MovieLens 100K que contiene u.data",
    )
    parser.add_argument("--user", help="Usuario para generar recomendaciones")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--factors", type=int, default=6)
    parser.add_argument("--population", type=int, default=25)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--mutation-sigma", type=float, default=0.08)
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.02,
        help="Fraccion de parametros que se mutan en cada hijo",
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.60,
        help="Probabilidad de recombinar dos padres elite",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Muestra el fitness durante el entrenamiento")
    parser.add_argument(
        "--explain-plots",
        action="store_true",
        help="Genera graficos de explicabilidad del entrenamiento y recomendaciones",
    )
    parser.add_argument(
        "--plots-dir",
        default="explanation_plots",
        help="Carpeta de salida para los graficos de explicabilidad",
    )
    parser.add_argument(
        "--user-explain-plots",
        action="store_true",
        help="Genera graficos explicativos orientados al usuario final",
    )
    parser.add_argument(
        "--user-plots-dir",
        default="user_explanation_plots",
        help="Carpeta de salida para los graficos orientados al usuario",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
