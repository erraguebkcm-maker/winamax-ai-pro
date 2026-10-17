"""Calculs d'équité (Monte-Carlo), cotes du pot et force préflop."""
import random
from .cards import evaluate


def chen_score(c1: int, c2: int) -> float:
    """Formule de Chen (Bill Chen) : force préflop approximative, -1 à 20."""
    r1, r2 = sorted([c1 // 4, c2 // 4], reverse=True)
    base = {12: 10, 11: 8, 10: 7, 9: 6}.get(r1, (r1 + 2) / 2)
    if r1 == r2:
        return max(base * 2, 5)
    s = base
    if c1 % 4 == c2 % 4:
        s += 2
    gap = r1 - r2 - 1
    s -= [0, 1, 2, 4][gap] if gap < 4 else 5
    if gap <= 1 and r1 < 10:
        s += 1
    return s


# Classement des 1326 mains de départ pour définir des « fourchettes » (top X %)
_ALL = [(a, b) for a in range(52) for b in range(a + 1, 52)]
_SORTED = sorted(_ALL, key=lambda h: chen_score(*h), reverse=True)
_PERCENTILE = {h: i / len(_SORTED) for i, h in enumerate(_SORTED)}


def hand_percentile(c1: int, c2: int) -> float:
    """0.0 = meilleure main de départ (AA), 1.0 = pire (72o)."""
    return _PERCENTILE[(min(c1, c2), max(c1, c2))]


def equity(hero: list[int], board: list[int], n_opp: int, iters: int = 300,
           opp_range: float = 1.0, rng: random.Random | None = None) -> float:
    """Probabilité de gagner (partages comptés au prorata) contre n_opp adversaires.

    opp_range : part des mains de départ jouées par l'adversaire (0.2 = top 20 %).
    """
    rng = rng or random
    if n_opp <= 0:
        return 1.0
    dead = set(hero) | set(board)
    deck = [c for c in range(52) if c not in dead]
    win = 0.0
    done = 0
    tries = 0
    while done < iters and tries < iters * 20:
        tries += 1
        rng.shuffle(deck)
        idx = 0
        opps = []
        ok = True
        for _ in range(n_opp):
            h = deck[idx:idx + 2]
            idx += 2
            if opp_range < 1.0 and hand_percentile(*h) > opp_range:
                ok = False
                break
            opps.append(h)
        if not ok:
            continue
        full = board + deck[idx:idx + 5 - len(board)]
        hs = evaluate(hero + full)
        best = hs
        n_best = 1
        lost = False
        for h in opps:
            s = evaluate(h + full)
            if s > hs:
                lost = True
                break
            if s == hs:
                n_best += 1
        if not lost:
            win += 1.0 / n_best
        done += 1
    return win / max(done, 1)


def pot_odds(to_call: float, pot: float) -> float:
    """Équité minimale nécessaire pour suivre de façon rentable."""
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def call_ev(eq: float, to_call: float, pot: float) -> float:
    """Espérance (en jetons) d'un call, en ignorant les mises futures."""
    return eq * pot - (1 - eq) * to_call
