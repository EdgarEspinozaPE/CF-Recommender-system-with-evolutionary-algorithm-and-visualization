"""
Factorización matricial con descenso de gradiente estocástico (línea base).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evolutionary_cf import EvolutionaryCollaborativeFilter, Individual

K, REG, INIT = 6, 0.001, 0.3


class SGDMatrixFactorization(EvolutionaryCollaborativeFilter):
    """SGD con la misma interfaz que el modelo evolutivo."""

    def __init__(self, n_users, n_items, factors=K, learning_rate=0.01,
                 epochs=30, regularization=REG, init_scale=INIT, seed=42,
                 rating_min=1.0, rating_max=5.0, verbose=False):
        super().__init__(n_users=n_users, n_items=n_items, factors=factors,
                         population_size=4, generations=1,
                         regularization=regularization, init_scale=init_scale,
                         rating_min=rating_min, rating_max=rating_max,
                         seed=seed, verbose=verbose)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, ratings):
        global_mean, ub, ib = self._baseline_terms(ratings)
        ind = Individual(
            user_factors=[[self.random.gauss(0.0, self.init_scale)
                           for _ in range(self.factors)] for _ in range(self.n_users)],
            item_factors=[[self.random.gauss(0.0, self.init_scale)
                           for _ in range(self.factors)] for _ in range(self.n_items)],
            user_bias=list(ub), item_bias=list(ib), global_mean=global_mean)

        data = list(ratings)
        lr, reg = self.learning_rate, self.regularization
        for epoch in range(self.epochs):
            self.random.shuffle(data)
            for u, i, r in data:
                err = r - self._raw_predict_with(ind, u, i)
                pu, qi = ind.user_factors[u], ind.item_factors[i]
                for f in range(self.factors):
                    puf, qif = pu[f], qi[f]
                    pu[f] = puf + lr * (err * qif - reg * puf)
                    qi[f] = qif + lr * (err * puf - reg * qif)
                ind.user_bias[u] += lr * (err - reg * ind.user_bias[u])
                ind.item_bias[i] += lr * (err - reg * ind.item_bias[i])
            ind.fitness = self._fitness(ind, ratings)
            self.history_.append(ind.fitness)
            self.generation_stats_.append(
                {"generation": float(epoch), "best": ind.fitness, "mean": ind.fitness,
                 "worst": ind.fitness, "fitness_std": 0.0, "mutation_sigma": 0.0})
        self.best_ = ind
        return self
