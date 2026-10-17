"""Moteur pronostics : backtest chronologique (walk-forward), bankroll fictive, prédictions à venir."""
import math
from collections import defaultdict
from datetime import datetime
import numpy as np
from .models import EloModel, PoissonModel, EnsembleModel, MarketModel, UniformModel, RecentForm
from .value import MARKETS, ODDS_COL, fair_probs, assess, stake_size, margin

OUTCOMES = ("H", "D", "A")


def _result(row):
    return "H" if row["hg"] > row["ag"] else "D" if row["hg"] == row["ag"] else "A"


def _odds_bucket(o):
    return "1.40–1.99" if o < 2 else "2.00–2.99" if o < 3 else "3.00–5.00"


class SportsEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.elo, self.pois = EloModel(), PoissonModel()
        self.ens = EnsembleModel(self.elo, self.pois)
        self.models = [self.ens, self.elo, self.pois, MarketModel(), UniformModel()]
        self.form = RecentForm()

    # ---------- une sélection ----------
    def _candidates(self, row, pred):
        out = []
        games_min = min(self.form.games[row["home"]], self.form.games[row["away"]])
        fair_1x2 = fair_probs([row["odds_h"], row["odds_d"], row["odds_a"]])
        fair_ou = fair_probs([row["odds_o25"], row["odds_u25"]])
        for mk in ("H", "D", "A", "O25", "U25"):
            fair = fair_1x2 if mk in OUTCOMES else fair_ou
            if fair is None:
                continue
            idx = OUTCOMES.index(mk) if mk in OUTCOMES else (0 if mk == "O25" else 1)
            odds = float(row[ODDS_COL[mk]])
            a = assess(pred[mk], fair[idx], odds, pred.get("_disagree", 0.0), games_min, self.cfg)
            a.update({"market": mk, "odds": odds, "p_model": pred[mk], "p_fair": fair[idx],
                      "p_implied": 1 / odds, "games_min": games_min})
            out.append(a)
        return out

    def _update(self, row):
        for m in (self.elo, self.pois) + tuple(self.models[3:]):
            m.update(row)
        self.form.update(row)

    # ---------- backtest ----------
    def backtest(self, hist, warmup=380):
        cfg = self.cfg
        scores = {m.name: {"brier": [], "logloss": []} for m in self.models}
        bank = cfg["bankroll_start"]
        peak, max_dd, stopped = bank, 0.0, False
        bets, curve, monthly = [], [], defaultdict(lambda: {"ens": [], "mkt": []})
        for i, row in hist.iterrows():
            preds = {m.name: m.predict(row) for m in self.models}
            if i >= warmup and preds["Marché (référence)"] is not None:
                y = _result(row)
                for name, p in preds.items():
                    if p is None:
                        continue
                    b = sum((p[k] - (1.0 if k == y else 0.0)) ** 2 for k in OUTCOMES)
                    scores[name]["brier"].append(b)
                    scores[name]["logloss"].append(-math.log(max(p[y], 1e-9)))
                mth = str(row["date"])[:7]
                monthly[mth]["ens"].append(scores[self.ens.name]["brier"][-1])
                monthly[mth]["mkt"].append(scores["Marché (référence)"]["brier"][-1])
                # paris fictifs
                if not stopped:
                    cands = [c for c in self._candidates(row, preds[self.ens.name]) if c["selected"]]
                    for c in sorted(cands, key=lambda c: -c["quality"])[:1]:
                        stake = stake_size(bank, c["p_model"], c["odds"], cfg)
                        if stake <= 0:
                            continue
                        won = (c["market"] == y) if c["market"] in OUTCOMES else \
                            ((row["hg"] + row["ag"] > 2.5) == (c["market"] == "O25"))
                        pnl = stake * (c["odds"] - 1) if won else -stake
                        bank += pnl
                        bets.append({"date": str(row["date"]), "match": f'{row["home"]} – {row["away"]}',
                                     "market": c["market"], "odds": c["odds"], "p": round(c["p_model"], 3),
                                     "ev": round(c["ev"], 3), "stake": stake, "won": won, "pnl": round(pnl, 2),
                                     "bank": round(bank, 2)})
                        peak = max(peak, bank)
                        max_dd = max(max_dd, (peak - bank) / peak)
                        if bank < cfg["bankroll_start"] * (1 - cfg["stop_loss_pct"]):
                            stopped = True
                curve.append({"date": str(row["date"]), "bank": round(bank, 2)})
            self._update(row)
        return self._summarize(scores, bets, bank, max_dd, stopped, curve, monthly)

    def _summarize(self, scores, bets, bank, max_dd, stopped, curve, monthly):
        cfg = self.cfg
        table = []
        for name, s in scores.items():
            if s["brier"]:
                table.append({"model": name, "brier": round(float(np.mean(s["brier"])), 4),
                              "logloss": round(float(np.mean(s["logloss"])), 4), "n": len(s["brier"])})
        table.sort(key=lambda r: r["brier"])
        staked = sum(b["stake"] for b in bets)
        pnl = sum(b["pnl"] for b in bets)
        rets = np.array([b["pnl"] / b["stake"] for b in bets]) if bets else np.array([])
        se = float(rets.std(ddof=1) / math.sqrt(len(rets))) if len(rets) > 1 else float("nan")
        roi = pnl / staked if staked else 0.0
        by = defaultdict(lambda: {"n": 0, "won": 0, "pnl": 0.0, "staked": 0.0})
        for b in bets:
            for key in (MARKETS[b["market"]], "Cotes " + _odds_bucket(b["odds"])):
                g = by[key]
                g["n"] += 1; g["won"] += b["won"]; g["pnl"] += b["pnl"]; g["staked"] += b["stake"]
        breakdown = [{"groupe": k, "paris": v["n"], "réussite": round(v["won"] / v["n"], 3),
                      "roi": round(v["pnl"] / v["staked"], 3) if v["staked"] else 0} for k, v in by.items()]
        worst = sorted(({"mois": m, "écart_brier": round(np.mean(v["ens"]) - np.mean(v["mkt"]), 4),
                         "matchs": len(v["ens"])} for m, v in monthly.items()),
                       key=lambda r: -r["écart_brier"])[:5]
        ens_b = next(r["brier"] for r in table if r["model"].startswith("Ensemble"))
        mkt_b = next((r["brier"] for r in table if r["model"].startswith("Marché")), None)
        lo, hi = roi - 1.96 * se, roi + 1.96 * se
        verdict = ("Le modèle bat le marché en précision ET le rendement est positif avec un intervalle "
                   "de confiance au-dessus de 0." if (mkt_b and ens_b < mkt_b and lo > 0) else
                   "Aucun avantage démontré : le modèle ne prouve pas qu'il fait mieux que les cotes du marché. "
                   "Les « paris à valeur » doivent être considérés comme NON fiables.")
        return {"models": table, "bets": bets, "n_bets": len(bets), "staked": round(staked, 2),
                "pnl": round(pnl, 2), "roi": round(roi, 4), "roi_ci95": [round(lo, 4), round(hi, 4)],
                "hit_rate": round(sum(b["won"] for b in bets) / len(bets), 3) if bets else 0,
                "bank_end": round(bank, 2), "max_drawdown": round(max_dd, 3), "stopped": stopped,
                "curve": curve[::max(1, len(curve) // 200)], "breakdown": breakdown,
                "worst_months": worst, "verdict": verdict}

    # ---------- matchs à venir ----------
    def predict_upcoming(self, upcoming, data_time: str):
        cards = []
        for _, row in upcoming.iterrows():
            pred = self.ens.predict(row)
            e, p = self.elo.predict(row), self.pois.predict(row)
            lh, la = self.pois.lambdas(row["home"], row["away"])
            cands = self._candidates(row, pred)
            m = margin([row["odds_h"], row["odds_d"], row["odds_a"]])
            for c in cands:
                pro, con = [], []
                fh, fa = self.form.form(row["home"]), self.form.form(row["away"])
                pro.append(f"Forme (5 derniers) : {row['home']} {fh} · {row['away']} {fa}")
                pro.append(f"Buts attendus (Poisson) : {lh:.2f} – {la:.2f}")
                pro.append(f"Elo dit {e[c['market']]:.0%}, Poisson dit {p[c['market']]:.0%}"
                           if c["market"] in OUTCOMES else f"Poisson : {p[c['market']]:.0%}")
                if c["games_min"] < self.cfg["min_games"]:
                    con.append(f"Seulement {c['games_min']} matchs connus pour une des équipes")
                if pred.get("_disagree", 0) > 0.06:
                    con.append("Les modèles Elo et Poisson ne sont pas d'accord")
                con.append("Blessures / compositions : NON disponibles dans cette source — à vérifier soi-même")
                if m is not None:
                    con.append(f"Marge du bookmaker sur ce match : {m:.1%}")
                c.update({"match": f"{row['home']} contre {row['away']}", "league": row["league"],
                          "date": str(row["date"]), "market_name": MARKETS[c["market"]],
                          "pro": pro, "con": con, "updated": data_time, "source": row["source"]})
                cards.append(c)
        cards.sort(key=lambda c: (-c["selected"], -c["quality"]))
        return cards
