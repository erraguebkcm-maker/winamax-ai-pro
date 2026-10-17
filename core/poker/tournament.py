"""Moteur de tournoi Texas Hold'em No Limit (Sit & Go simulé, jetons fictifs).

Gère : blindes croissantes, tours d'enchères complets, tapis, pots annexes, abattage.
"""
import random
from dataclasses import dataclass
from .cards import new_deck, evaluate
from .players import (AutonomousAgent, AgentParams, OpponentStats, make_opponent,
                      OPPONENT_STYLES)

BLINDS = [(10, 20), (15, 30), (25, 50), (50, 100), (75, 150), (100, 200),
          (150, 300), (200, 400), (300, 600), (400, 800), (600, 1200), (1000, 2000)]


@dataclass
class Seat:
    player: object
    stack: float
    hole: list = None
    bet: float = 0.0       # mise sur la rue en cours
    contrib: float = 0.0   # total misé sur la main
    folded: bool = False
    allin: bool = False


class Tournament:
    def __init__(self, players, start_stack=1500, hands_per_level=10, seed=0, max_hands=500):
        self.rng = random.Random(seed)
        self.seats = [Seat(p, start_stack) for p in players]
        self.hands_per_level = hands_per_level
        self.max_hands = max_hands
        self.button = 0
        self.hand_no = 0
        self.eliminated: list[str] = []  # ordre d'élimination
        self.stats: dict[str, OpponentStats] = {}
        for p in players:
            self.stats[p.name] = OpponentStats()
            if isinstance(p, AutonomousAgent):
                p.stats = self.stats

    # ---------- utilitaires ----------
    def alive(self):
        return [s for s in self.seats if s.stack > 0]

    def blinds(self):
        lvl = min(self.hand_no // self.hands_per_level, len(BLINDS) - 1)
        return BLINDS[lvl]

    def _put(self, seat, amount):
        amount = min(amount, seat.stack)
        seat.stack -= amount
        seat.bet += amount
        seat.contrib += amount
        if seat.stack <= 0:
            seat.allin = True
        return amount

    # ---------- une main ----------
    def play_hand(self):
        players = self.alive()
        n = len(players)
        sb, bb = self.blinds()
        deck = new_deck(self.rng)
        for s in players:
            s.hole = [deck.pop(), deck.pop()]
            s.bet = s.contrib = 0.0
            s.folded = s.allin = False
            self.stats[s.player.name].hands += 1
        self.button %= n
        if n == 2:
            sb_i, bb_i = self.button, (self.button + 1) % n
        else:
            sb_i, bb_i = (self.button + 1) % n, (self.button + 2) % n
        self._put(players[sb_i], sb)
        self._put(players[bb_i], bb)
        board: list[int] = []
        vpip_done = set()
        first = (bb_i + 1) % n
        for street in range(4):
            if street > 0:
                for s in players:
                    s.bet = 0.0
                deck.pop()  # carte brûlée
                board += [deck.pop() for _ in range(3 if street == 1 else 1)]
                first = (self.button + 1) % n if n > 2 else bb_i
            self._betting_round(players, board, street, first, bb, vpip_done)
            if sum(not s.folded for s in players) == 1:
                break
        self._showdown(players, board, deck)
        for s in players:
            if s.stack <= 0:
                self.eliminated.append(s.player.name)
        self.button = (self.button + 1) % max(len(self.alive()), 1)
        self.hand_no += 1

    def _betting_round(self, players, board, street, first, bb, vpip_done):
        n = len(players)
        current = max(s.bet for s in players)
        last_raise = bb
        order = [(first + k) % n for k in range(n)]
        to_act = [i for i in order if not players[i].folded and not players[i].allin]
        while to_act:
            if sum(not s.folded for s in players) == 1:
                return
            i = to_act.pop(0)
            s = players[i]
            if s.folded or s.allin:
                continue
            pot = sum(x.contrib for x in players)
            state = {
                "hole": s.hole, "board": list(board), "street": street, "pot": pot,
                "to_call": min(current - s.bet, s.stack), "current_bet": current,
                "my_bet": s.bet, "min_raise_to": current + last_raise, "stack": s.stack,
                "bb": bb, "position": (i - self.button) % n,
                "active_opponents": [p.player.name for p in players if p is not s and not p.folded],
            }
            action, amount = s.player.decide(state)
            st = self.stats[s.player.name]
            if action == "fold" and state["to_call"] > 0:
                s.folded = True
                continue
            if action in ("check", "fold"):
                continue
            if action == "call":
                self._put(s, current - s.bet)
                st.passive += 1
            elif action == "raise":
                target = min(amount, s.bet + s.stack)
                if target <= current:
                    self._put(s, current - s.bet)
                    st.passive += 1
                else:
                    self._put(s, target - s.bet)
                    if s.bet - current >= last_raise:
                        last_raise = s.bet - current
                    current = s.bet
                    st.aggressive += 1
                    idx = order.index(i)
                    to_act = [j for j in order[idx + 1:] + order[:idx]
                              if not players[j].folded and not players[j].allin]
            if street == 0 and s.player.name not in vpip_done:
                vpip_done.add(s.player.name)
                st.vpip += 1

    def _showdown(self, players, board, deck):
        live = [s for s in players if not s.folded]
        if len(live) == 1:
            live[0].stack += sum(s.contrib for s in players)
            return
        while len(board) < 5:
            board.append(deck.pop())
        scores = {id(s): evaluate(s.hole + board) for s in live}
        levels = sorted({s.contrib for s in players if s.contrib > 0})
        prev = 0.0
        for lvl in levels:
            part = sum(min(s.contrib, lvl) - min(s.contrib, prev) for s in players)
            elig = [s for s in live if s.contrib >= lvl]
            if not elig:  # ne devrait pas arriver ; rendre au plus gros contributeur vivant
                elig = live
            best = max(scores[id(s)] for s in elig)
            winners = [s for s in elig if scores[id(s)] == best]
            for w in winners:
                w.stack += part / len(winners)
            prev = lvl

    # ---------- tournoi complet ----------
    def run(self) -> list[str]:
        while len(self.alive()) > 1 and self.hand_no < self.max_hands:
            self.play_hand()
        rest = sorted(self.alive(), key=lambda s: s.stack)
        ranking = [s.player.name for s in rest] if self.hand_no >= self.max_hands else \
            [s.player.name for s in rest]
        order = self.eliminated + ranking   # du 1er éliminé au vainqueur
        return list(reversed(order))         # 1re place en tête


PAYOUTS_6MAX = [0.65, 0.35, 0, 0, 0, 0]


def play_sng(seed: int, params: AgentParams | None = None, buy_in: float = 10.0,
             opponents: list[str] | None = None) -> dict:
    """Joue un Sit & Go 6 joueurs : robot IA + 5 adversaires virtuels. Jetons fictifs."""
    rng = random.Random(seed)
    styles = opponents or [rng.choice(OPPONENT_STYLES) for _ in range(5)]
    agent = AutonomousAgent("Robot IA", random.Random(seed + 1), params)
    opps = [make_opponent(st, f"Bot{i + 1} · {st.split(' ')[0]}", random.Random(seed * 7 + i))
            for i, st in enumerate(styles)]
    t = Tournament([agent] + opps, seed=seed)
    ranking = t.run()
    place = ranking.index("Robot IA") + 1
    prize = PAYOUTS_6MAX[place - 1] * buy_in * 6
    acts = [d["action"] for d in agent.log]
    return {"seed": seed, "place": place, "prize": prize, "profit": prize - buy_in,
            "hands": t.hand_no, "opponents": styles,
            "actions": {a: acts.count(a) for a in set(acts)}}
