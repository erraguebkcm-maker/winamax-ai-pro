"""Paris « à valeur » : marge du bookmaker, espérance, qualité statistique, gestion du risque."""
import math

MARKETS = {"H": "Victoire domicile", "D": "Match nul", "A": "Victoire extérieur",
           "O25": "Plus de 2,5 buts", "U25": "Moins de 2,5 buts"}
ODDS_COL = {"H": "odds_h", "D": "odds_d", "A": "odds_a", "O25": "odds_o25", "U25": "odds_u25"}


def _ok(o):
    return o is not None and not (isinstance(o, float) and math.isnan(o)) and o > 1.0


def margin(odds) -> float | None:
    if not all(_ok(o) for o in odds):
        return None
    return sum(1 / o for o in odds) - 1


def fair_probs(odds, method="power"):
    """Probabilités « justes » sans la marge du bookmaker.
    'power' corrige mieux le biais favori/outsider que la simple normalisation."""
    if not all(_ok(o) for o in odds):
        return None
    imp = [1 / o for o in odds]
    if method == "proportional":
        s = sum(imp)
        return [p / s for p in imp]
    lo, hi = 0.5, 3.0          # cherche k tel que somme(p_i^k) = 1
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(p ** k for p in imp) > 1:
            lo = k
        else:
            hi = k
    return [p ** k for p in imp]


def expected_value(p: float, odds: float) -> float:
    """Gain moyen théorique par euro misé (0.05 = +5 %)."""
    return p * odds - 1


def assess(p_model, p_fair, odds, disagree, games_min, cfg) -> dict:
    """Évalue une sélection. Renvoie EV, avantage, incertitude, qualité et décision."""
    ev = expected_value(p_model, odds)
    edge = p_model - p_fair
    # incertitude : désaccord des modèles + manque de données + cotes élevées (plus de variance)
    unc = 0.5 * disagree + 0.06 * math.exp(-games_min / 10) + 0.01 * max(odds - 2, 0)
    level = "faible" if unc < 0.04 else "moyenne" if unc < 0.08 else "élevée"
    quality = edge / max(unc, 0.01)
    reasons = []
    if games_min < cfg["min_games"]:
        reasons.append(f"trop peu de matchs connus ({games_min})")
    if not (cfg["min_odds"] <= odds <= cfg["max_odds"]):
        reasons.append("cote hors de la plage étudiée")
    if ev < cfg["min_ev"]:
        reasons.append("espérance insuffisante")
    if unc >= cfg["max_uncertainty"]:
        reasons.append("incertitude trop élevée")
    if edge > cfg["suspicious_edge"]:
        reasons.append("écart suspect avec le marché (info manquante probable : blessure, compo…)")
    return {"ev": ev, "edge": edge, "uncertainty": unc, "unc_level": level,
            "quality": quality, "selected": not reasons, "rejected_because": reasons}


def stake_size(bankroll: float, p: float, odds: float, cfg) -> float:
    """Mise fictive : quart de Kelly, plafonnée. Jamais de martingale ni de « refaire ses pertes »."""
    b = odds - 1
    kelly = max((p * b - (1 - p)) / b, 0.0)
    return round(min(kelly * cfg["kelly_fraction"], cfg["max_stake_pct"]) * bankroll, 2)
