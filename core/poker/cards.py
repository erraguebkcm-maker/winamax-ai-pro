"""Cartes, paquet et évaluateur de mains Texas Hold'em (5 à 7 cartes).

Une carte = entier 0..51 : rang = c // 4 (0 = deux ... 12 = as), couleur = c % 4.
evaluate() renvoie un tuple comparable : plus grand = meilleure main.
"""
from collections import Counter
import random

RANKS = "23456789TJQKA"
SUITS = "shdc"
CATEGORIES = ["Hauteur", "Paire", "Double paire", "Brelan", "Quinte",
              "Couleur", "Full", "Carré", "Quinte flush"]


def card(s: str) -> int:
    """'Ah' -> entier."""
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def to_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


def new_deck(rng: random.Random | None = None) -> list[int]:
    d = list(range(52))
    (rng or random).shuffle(d)
    return d


def _straight_high(ranks: set[int]) -> int:
    """Rang haut de la meilleure quinte, ou -1."""
    rs = set(ranks)
    if 12 in rs:
        rs.add(-1)  # as bas (A-2-3-4-5)
    for hi in range(12, 2, -1):
        if all((hi - k) in rs for k in range(5)):
            return hi
    return -1


def evaluate(cards: list[int]) -> tuple:
    ranks = [c // 4 for c in cards]
    suits = [c % 4 for c in cards]
    # Couleur / quinte flush
    sc = Counter(suits)
    flush_suit = next((s for s, n in sc.items() if n >= 5), None)
    if flush_suit is not None:
        fr = [c // 4 for c in cards if c % 4 == flush_suit]
        sf = _straight_high(set(fr))
        if sf >= 0:
            return (8, sf)
    cnt = Counter(ranks)
    groups = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if groups[0][1] == 4:
        quad = groups[0][0]
        kick = max(r for r in ranks if r != quad)
        return (7, quad, kick)
    trips = sorted([r for r, n in cnt.items() if n == 3], reverse=True)
    pairs = sorted([r for r, n in cnt.items() if n == 2], reverse=True)
    if trips and (len(trips) > 1 or pairs):
        second = max(trips[1:] + pairs)
        return (6, trips[0], second)
    if flush_suit is not None:
        fr = sorted([c // 4 for c in cards if c % 4 == flush_suit], reverse=True)[:5]
        return (5, *fr)
    st = _straight_high(set(ranks))
    if st >= 0:
        return (4, st)
    if trips:
        kick = sorted([r for r in ranks if r != trips[0]], reverse=True)[:2]
        return (3, trips[0], *kick)
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kick = max(r for r in ranks if r not in (p1, p2))
        return (2, p1, p2, kick)
    if pairs:
        kick = sorted([r for r in ranks if r != pairs[0]], reverse=True)[:3]
        return (1, pairs[0], *kick)
    return (0, *sorted(ranks, reverse=True)[:5])


def category(score: tuple) -> str:
    return CATEGORIES[score[0]]
