"""Tests automatisés. Lancer :  python -m unittest -v"""
import random
import unittest

from config import SETTINGS
from core.poker.cards import card, evaluate
from core.poker.equity import equity, pot_odds, hand_percentile
from core.poker.players import AutonomousAgent, make_opponent, OPPONENT_STYLES
from core.poker.tournament import Tournament, play_sng
from core.poker.trainer import make_exercise
from core.sports.data import make_demo_league
from core.sports.engine import SportsEngine
from core.sports.value import fair_probs, expected_value, stake_size, margin


def h(*cs):
    return [card(c) for c in cs]


class TestEvaluator(unittest.TestCase):
    def test_ordre_des_mains(self):
        royal = evaluate(h("Ah", "Kh", "Qh", "Jh", "Th", "2c", "3d"))
        quads = evaluate(h("9h", "9d", "9s", "9c", "Kd", "2c", "3d"))
        full = evaluate(h("9h", "9d", "9s", "Kc", "Kd", "2c", "3d"))
        flush = evaluate(h("2h", "7h", "9h", "Jh", "Kh", "2c", "3d"))
        straight = evaluate(h("5c", "6d", "7h", "8s", "9c", "Kd", "2h"))
        wheel = evaluate(h("Ac", "2d", "3h", "4s", "5c", "Kd", "9h"))
        self.assertTrue(royal > quads > full > flush > straight > wheel)
        self.assertEqual(wheel, (4, 3))

    def test_kicker(self):
        self.assertGreater(evaluate(h("Ah", "Ad", "Kc", "7s", "4d")), evaluate(h("As", "Ac", "Qc", "7h", "4c")))


class TestEquity(unittest.TestCase):
    def test_paire_as(self):
        eq = equity(h("Ah", "Ad"), [], 1, iters=3000, rng=random.Random(1))
        self.assertAlmostEqual(eq, 0.85, delta=0.03)

    def test_cotes_du_pot(self):
        self.assertAlmostEqual(pot_odds(50, 100), 1 / 3)

    def test_classement_preflop(self):
        self.assertLess(hand_percentile(*h("Ah", "Ad")), 0.01)
        self.assertGreater(hand_percentile(*h("7c", "2d")), 0.9)


class TestTournoi(unittest.TestCase):
    def test_conservation_des_jetons(self):
        rng = random.Random(3)
        players = [AutonomousAgent("Robot IA", random.Random(4))] + \
                  [make_opponent(s, f"B{i}", random.Random(i)) for i, s in enumerate(OPPONENT_STYLES)]
        t = Tournament(players, seed=5)
        total = sum(s.stack for s in t.seats)
        for _ in range(60):
            if len(t.alive()) < 2:
                break
            t.play_hand()
            self.assertAlmostEqual(sum(s.stack for s in t.seats), total, places=6)
            self.assertTrue(all(s.stack >= -1e-9 for s in t.seats))

    def test_reproductible(self):
        self.assertEqual(play_sng(11)["place"], play_sng(11)["place"])

    def test_exercice(self):
        ex = make_exercise(1)
        self.assertIn(ex["reponse"], ("SUIVRE", "SE COUCHER"))
        self.assertEqual(ex["reponse"] == "SUIVRE", ex["equite"] > ex["besoin"])


class TestValeur(unittest.TestCase):
    def test_marge_retiree(self):
        odds = [1.9, 3.5, 4.2]
        self.assertGreater(margin(odds), 0)
        for m in ("power", "proportional"):
            self.assertAlmostEqual(sum(fair_probs(odds, m)), 1.0, places=6)
        self.assertIsNone(fair_probs([2.0, float("nan"), 3.0]))

    def test_esperance(self):
        self.assertAlmostEqual(expected_value(0.5, 2.2), 0.1)

    def test_mise_plafonnee_et_sans_martingale(self):
        self.assertLessEqual(stake_size(1000, 0.9, 3.0, SETTINGS), 1000 * SETTINGS["max_stake_pct"])
        self.assertEqual(stake_size(1000, 0.3, 2.0, SETTINGS), 0)  # pas d'avantage → pas de mise
        # après une perte (bankroll plus petite), la mise baisse : jamais de « refaire ses pertes »
        self.assertLess(stake_size(900, 0.55, 2.0, SETTINGS), stake_size(1000, 0.55, 2.0, SETTINGS))


class TestSansFuite(unittest.TestCase):
    def test_pas_d_information_future(self):
        """La prédiction d'un match ne doit pas changer si on modifie les matchs SUIVANTS."""
        hist, _ = make_demo_league(seed=7)
        cut = 500
        a, b = SportsEngine(SETTINGS), SportsEngine(SETTINGS)
        altered = hist.copy()
        altered.loc[cut + 1:, "hg"] = 9
        for eng, df in ((a, hist), (b, altered)):
            for _, row in df.iloc[:cut].iterrows():
                eng._update(row)
        row = hist.iloc[cut]
        self.assertEqual(a.ens.predict(row), b.ens.predict(row))

    def test_backtest_complet(self):
        hist, up = make_demo_league(seed=1)
        eng = SportsEngine(SETTINGS)
        r = eng.backtest(hist)
        names = [m["model"] for m in r["models"]]
        self.assertIn("Marché (référence)", names)
        self.assertIn("Naïf (fréquences)", names)
        cards = eng.predict_upcoming(up, "test")
        self.assertTrue(all(0 < c["p_model"] < 1 for c in cards))


if __name__ == "__main__":
    unittest.main()
