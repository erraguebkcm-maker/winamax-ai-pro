"""Modèles de prédiction football (1X2 + plus/moins de 2,5 buts).

Tous les modèles sont « en ligne » : ils prédisent un match AVANT de voir son résultat,
puis se mettent à jour. Impossible d'utiliser une information future par accident.
"""
import math
from collections import defaultdict, deque
import numpy as np


def poisson_matrix(lh, la, maxg=10):
    k = np.arange(maxg + 1)
    fact = np.array([math.factorial(int(x)) for x in k], dtype=float)
    ph = np.exp(-lh) * lh ** k / fact
    pa = np.exp(-la) * la ** k / fact
    return np.outer(ph, pa)


def matrix_probs(m):
    total = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return {"H": float(np.tril(m, -1).sum()), "D": float(np.trace(m)), "A": float(np.triu(m, 1).sum()),
            "O25": float(m[total > 2.5].sum()), "U25": float(m[total < 2.5].sum())}


class MarketModel:
    """Référence : probabilités implicites des bookmakers, marge retirée."""
    name = "Marché (référence)"

    def predict(self, row):
        from .value import fair_probs
        p = fair_probs([row["odds_h"], row["odds_d"], row["odds_a"]])
        if p is None:
            return None
        out = {"H": p[0], "D": p[1], "A": p[2]}
        ou = fair_probs([row["odds_o25"], row["odds_u25"]])
        if ou is not None:
            out.update({"O25": ou[0], "U25": ou[1]})
        return out

    def update(self, row):
        pass


class UniformModel:
    """Référence naïve : fréquences historiques du championnat."""
    name = "Naïf (fréquences)"

    def __init__(self):
        self.c = {"H": 1.0, "D": 1.0, "A": 1.0, "O25": 1.0, "U25": 1.0}

    def predict(self, row):
        s = self.c["H"] + self.c["D"] + self.c["A"]
        t = self.c["O25"] + self.c["U25"]
        return {"H": self.c["H"] / s, "D": self.c["D"] / s, "A": self.c["A"] / s,
                "O25": self.c["O25"] / t, "U25": self.c["U25"] / t}

    def update(self, row):
        r = "H" if row["hg"] > row["ag"] else "D" if row["hg"] == row["ag"] else "A"
        self.c[r] += 1
        self.c["O25" if row["hg"] + row["ag"] > 2.5 else "U25"] += 1


class EloModel:
    """Elo avec avantage du terrain et prise en compte de l'écart de buts."""
    name = "Elo"

    def __init__(self, k=20.0, home_adv=65.0):
        self.k, self.h = k, home_adv
        self.r = defaultdict(lambda: 1500.0)
        self.draw_rate = 0.26
        self.n = 0

    def expected(self, home, away):
        return 1 / (1 + 10 ** ((self.r[away] - self.r[home] - self.h) / 400))

    def predict(self, row):
        e = self.expected(row["home"], row["away"])
        pd_ = self.draw_rate * (1 - 1.6 * (e - 0.5) ** 2) * 1.0
        pd_ = min(max(pd_, 0.12), 0.34)
        ph = max(e - pd_ / 2, 0.02)
        pa = max(1 - e - pd_ / 2, 0.02)
        s = ph + pd_ + pa
        return {"H": ph / s, "D": pd_ / s, "A": pa / s}

    def update(self, row):
        e = self.expected(row["home"], row["away"])
        res = 1.0 if row["hg"] > row["ag"] else 0.5 if row["hg"] == row["ag"] else 0.0
        mult = math.log(abs(row["hg"] - row["ag"]) + 1) + 1
        delta = self.k * mult * (res - e)
        self.r[row["home"]] += delta
        self.r[row["away"]] -= delta
        self.n += 1
        self.draw_rate += 0.01 * ((1.0 if res == 0.5 else 0.0) - self.draw_rate)


class PoissonModel:
    """Forces d'attaque/défense (moyennes pondérées, les matchs récents comptent plus),
    avec retour vers la moyenne quand une équipe a peu joué (approche bayésienne simple)."""
    name = "Poisson"

    def __init__(self, decay=0.985, prior_games=6.0):
        self.decay, self.prior = decay, prior_games
        self.att = defaultdict(lambda: [0.0, 0.0])   # [somme pondérée buts / moyenne, poids]
        self.dfn = defaultdict(lambda: [0.0, 0.0])
        self.home_g, self.away_g, self.w = 1.5, 1.2, 1.0
        self.games = defaultdict(int)

    def _strength(self, d, team):
        s, w = d[team]
        return (s + self.prior) / (w + self.prior)  # retour vers 1.0

    def lambdas(self, home, away):
        lh = self.home_g * self._strength(self.att, home) * self._strength(self.dfn, away)
        la = self.away_g * self._strength(self.att, away) * self._strength(self.dfn, home)
        return lh, la

    def predict(self, row):
        lh, la = self.lambdas(row["home"], row["away"])
        return matrix_probs(poisson_matrix(lh, la))

    def update(self, row):
        a = 0.02
        self.home_g += a * (row["hg"] - self.home_g)
        self.away_g += a * (row["ag"] - self.away_g)
        for team, scored, conceded, avg_s, avg_c in (
                (row["home"], row["hg"], row["ag"], self.home_g, self.away_g),
                (row["away"], row["ag"], row["hg"], self.away_g, self.home_g)):
            for d, val, avg in ((self.att, scored, avg_s), (self.dfn, conceded, avg_c)):
                s, w = d[team]
                d[team] = [s * self.decay + val / avg, w * self.decay + 1]
            self.games[team] += 1


class EnsembleModel:
    """Moyenne Elo + Poisson pour le 1X2, Poisson seul pour les buts."""
    name = "Ensemble (Elo+Poisson)"

    def __init__(self, elo: EloModel, pois: PoissonModel, w_elo=0.4):
        self.elo, self.pois, self.w = elo, pois, w_elo

    def predict(self, row):
        e, p = self.elo.predict(row), self.pois.predict(row)
        out = {k: self.w * e[k] + (1 - self.w) * p[k] for k in ("H", "D", "A")}
        out.update({"O25": p["O25"], "U25": p["U25"]})
        out["_disagree"] = max(abs(e[k] - p[k]) for k in ("H", "D", "A"))
        return out

    def update(self, row):
        pass  # les sous-modèles sont mis à jour séparément


class RecentForm:
    """Forme récente et nombre de matchs disponibles (pour les arguments et la fiabilité)."""

    def __init__(self, n=5):
        self.last = defaultdict(lambda: deque(maxlen=n))
        self.games = defaultdict(int)

    def update(self, row):
        for t, gf, ga in ((row["home"], row["hg"], row["ag"]), (row["away"], row["ag"], row["hg"])):
            self.last[t].append("V" if gf > ga else "N" if gf == ga else "D")
            self.games[t] += 1

    def form(self, team):
        return "".join(self.last[team]) or "—"

    def points(self, team):
        return sum({"V": 3, "N": 1, "D": 0}[x] for x in self.last[team])
