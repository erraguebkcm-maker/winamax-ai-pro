"""Joueurs virtuels (styles variés) et agent IA autonome.

Chaque joueur reçoit un état de jeu (dict) et renvoie (action, montant_total_de_la_mise).
Actions : 'fold', 'check', 'call', 'raise' (montant = mise totale visée sur la rue).
Tout fonctionne UNIQUEMENT dans le simulateur interne : aucun lien avec un logiciel réel.
"""
import random
from dataclasses import dataclass, field
from .equity import equity, hand_percentile, pot_odds


@dataclass
class OpponentStats:
    """Statistiques observées sur un adversaire (modélisation)."""
    hands: int = 0
    vpip: int = 0          # a mis de l'argent volontairement préflop
    aggressive: int = 0    # mises / relances
    passive: int = 0       # suivis

    @property
    def vpip_rate(self) -> float:
        # a priori bayésien : 25 % sur 10 mains fictives
        return (self.vpip + 2.5) / (self.hands + 10)

    @property
    def aggression(self) -> float:
        return (self.aggressive + 1) / (self.passive + 1)


class Player:
    style = "base"

    def __init__(self, name: str, rng: random.Random):
        self.name = name
        self.rng = rng

    def decide(self, s: dict) -> tuple[str, float]:
        raise NotImplementedError

    # Outils communs
    def _raise_to(self, s, amount):
        amount = max(amount, s["min_raise_to"])
        amount = min(amount, s["stack"] + s["my_bet"])  # tapis max
        if amount <= s["current_bet"]:
            return self._call(s)
        return ("raise", amount)

    def _call(self, s):
        return ("call", 0) if s["to_call"] > 0 else ("check", 0)

    def _fold(self, s):
        return ("fold", 0) if s["to_call"] > 0 else ("check", 0)


class StyleBot(Player):
    """Adversaire paramétrable : serré/large, passif/agressif."""

    def __init__(self, name, rng, style, play_range, aggr, bluff, sticky):
        super().__init__(name, rng)
        self.style = style
        self.play_range = play_range  # part des mains jouées préflop
        self.aggr = aggr              # probabilité de relancer une main forte
        self.bluff = bluff            # fréquence de bluff
        self.sticky = sticky          # tendance à suivre trop souvent

    def decide(self, s):
        r = self.rng.random()
        if s["street"] == 0:
            pct = hand_percentile(*s["hole"])
            if pct > self.play_range:
                return self._fold(s) if r > self.bluff * 0.3 else self._raise_to(s, s["current_bet"] * 3 or s["bb"] * 3)
            if pct < self.play_range * 0.35 and r < self.aggr:
                return self._raise_to(s, max(s["current_bet"] * 3, s["bb"] * 3))
            if s["to_call"] > s["stack"] * 0.4 and pct > self.play_range * 0.3:
                return self._fold(s)
            return self._call(s)
        eq = equity(s["hole"], s["board"], 1, iters=50, rng=self.rng)
        need = pot_odds(s["to_call"], s["pot"])
        if eq > 0.7 and r < self.aggr:
            return self._raise_to(s, s["current_bet"] + s["pot"] * 0.75)
        if r < self.bluff and s["to_call"] == 0:
            return self._raise_to(s, s["pot"] * 0.6)
        if eq + self.sticky > need:
            return self._call(s)
        return self._fold(s)


def make_opponent(style: str, name: str, rng) -> StyleBot:
    presets = {
        "Roc (serré-passif)":        dict(play_range=0.12, aggr=0.25, bluff=0.02, sticky=0.0),
        "TAG (serré-agressif)":      dict(play_range=0.20, aggr=0.75, bluff=0.08, sticky=0.0),
        "Maniaque (large-agressif)": dict(play_range=0.60, aggr=0.85, bluff=0.30, sticky=0.05),
        "Station (suit tout)":       dict(play_range=0.55, aggr=0.15, bluff=0.03, sticky=0.30),
        "Débutant (aléatoire)":      dict(play_range=0.45, aggr=0.40, bluff=0.15, sticky=0.15),
    }
    return StyleBot(name, rng, style, **presets[style])


OPPONENT_STYLES = ["Roc (serré-passif)", "TAG (serré-agressif)", "Maniaque (large-agressif)",
                   "Station (suit tout)", "Débutant (aléatoire)"]


@dataclass
class AgentParams:
    """Paramètres du robot — c'est ce que l'optimiseur fait évoluer."""
    open_range: float = 0.22        # part des mains ouvertes préflop
    push_fold_bb: float = 12.0      # sous ce tapis (en grosses blindes) : tapis ou parole
    push_range_short: float = 0.35  # mains poussées à tapis court
    value_raise_eq: float = 0.68    # équité pour miser/relancer pour la valeur
    call_margin: float = 0.03       # marge de sécurité au-dessus des cotes du pot
    bluff_freq: float = 0.06
    iters: int = 250                # précision Monte-Carlo


class AutonomousAgent(Player):
    """Robot IA : équité Monte-Carlo + cotes du pot + modèle des adversaires + push/fold."""
    style = "Robot IA"

    def __init__(self, name, rng, params: AgentParams | None = None, stats: dict | None = None):
        super().__init__(name, rng)
        self.p = params or AgentParams()
        self.stats = stats if stats is not None else {}
        self.log: list[dict] = []  # journal des décisions (pour l'analyse)

    def _opp_range(self, s) -> float:
        """Estime la fourchette des adversaires encore en jeu d'après leur VPIP observé."""
        rates = [self.stats[o].vpip_rate for o in s["active_opponents"] if o in self.stats]
        if not rates:
            return 1.0
        r = min(1.0, sum(rates) / len(rates) * (1.0 if s["to_call"] == 0 else 0.8))
        return max(0.08, r)

    def decide(self, s):
        p = self.p
        bb = s["bb"]
        eff_bb = s["stack"] / bb
        n_opp = len(s["active_opponents"])
        r = self.rng.random()
        if s["street"] == 0:
            pct = hand_percentile(*s["hole"])
            # Mode tapis court : push/fold (théorie des tournois)
            if eff_bb <= p.push_fold_bb:
                limit = p.push_range_short * (1.3 if s["to_call"] <= bb else 0.7)
                act = ("raise", s["stack"] + s["my_bet"]) if pct <= limit else self._fold(s)
                self._record(s, act, None, "push/fold")
                return act
            facing_raise = s["current_bet"] > bb
            if pct <= p.open_range * (0.45 if facing_raise else 1.0):
                size = s["current_bet"] * 3 if facing_raise else bb * 2.5
                act = self._raise_to(s, size)
            elif facing_raise and pct <= p.open_range * 0.9 and s["to_call"] < s["stack"] * 0.15:
                act = self._call(s)
            elif s["to_call"] == 0:
                act = ("check", 0)
            elif s["to_call"] <= bb / 2 and pct <= 0.5:
                act = self._call(s)  # complète la petite blinde
            else:
                act = self._fold(s)
            self._record(s, act, None, "préflop")
            return act

        opp_range = self._opp_range(s)
        eq = equity(s["hole"], s["board"], n_opp, iters=p.iters, opp_range=opp_range, rng=self.rng)
        need = pot_odds(s["to_call"], s["pot"])
        if eq >= p.value_raise_eq:
            act = self._raise_to(s, s["current_bet"] + s["pot"] * 0.66)
        elif s["to_call"] == 0 and r < p.bluff_freq and n_opp == 1:
            act = self._raise_to(s, s["pot"] * 0.5)
        elif eq >= need + p.call_margin:
            act = self._call(s)
        else:
            act = self._fold(s)
        self._record(s, act, eq, f"équité {eq:.0%} vs besoin {need:.0%}")
        return act

    def _record(self, s, act, eq, reason):
        if len(self.log) < 5000:
            self.log.append({"street": s["street"], "action": act[0], "equity": eq,
                             "to_call": s["to_call"], "pot": s["pot"], "reason": reason})
