"""Entraîneur hors partie : exercices, comparaison de stratégies, auto-amélioration du robot.

Usage strictement HORS PARTIE : aucune fonction ne lit ni ne pilote un logiciel de poker réel.
"""
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from statistics import mean, stdev
from .cards import to_str, evaluate, category
from .equity import equity, pot_odds, call_ev
from .players import AgentParams
from .tournament import play_sng


# ---------- Exercices ----------
def make_exercise(seed: int) -> dict:
    """Exercice « suivre ou se coucher ? » avec correction chiffrée."""
    rng = random.Random(seed)
    deck = list(range(52))
    rng.shuffle(deck)
    hero, board = deck[:2], deck[2:2 + rng.choice([3, 4])]
    pot = rng.choice([200, 400, 600, 900, 1500])
    to_call = round(pot * rng.choice([0.33, 0.5, 0.75, 1.0]))
    eq = equity(hero, board, 1, iters=2500, rng=random.Random(seed))
    need = pot_odds(to_call, pot)
    ev = call_ev(eq, to_call, pot)
    good = "SUIVRE" if eq > need else "SE COUCHER"
    return {
        "main": " ".join(map(to_str, hero)), "tableau": " ".join(map(to_str, board)),
        "pot": pot, "a_suivre": to_call, "question": f"Pot {pot}, mise adverse {to_call}. Suivre ?",
        "equite": round(eq, 3), "besoin": round(need, 3), "ev_jetons": round(ev, 1),
        "reponse": good, "main_actuelle": category(evaluate(hero + board)),
        "explication": (f"Tu gagnes environ {eq:.0%} du temps contre une main au hasard. "
                        f"Pour suivre {to_call} et gagner un pot de {pot + to_call}, il faut au moins "
                        f"{need:.0%}. Espérance du call ≈ {ev:+.0f} jetons → {good}."),
    }


# ---------- Comparaison reproductible ----------
def _run(args):
    seed, params = args
    return play_sng(seed, AgentParams(**params))


def evaluate_params(params: AgentParams, seeds: list[int], workers: int = 4) -> dict:
    """Joue les mêmes tournois (mêmes graines) → comparaison équitable."""
    jobs = [(s, asdict(params)) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_run, jobs))
    else:
        res = [_run(j) for j in jobs]
    profits = [r["profit"] for r in res]
    return {"roi": mean(profits) / 10.0, "itm": sum(r["place"] <= 2 for r in res) / len(res),
            "wins": sum(r["place"] == 1 for r in res) / len(res),
            "sd": stdev(profits) if len(profits) > 1 else 0.0, "n": len(res), "results": res}


def improve(params: AgentParams, train_seeds, test_seeds, rounds=3, workers=4, rng_seed=0):
    """Auto-amélioration prudente : on garde un changement seulement s'il améliore le ROI
    sur les tournois d'entraînement, puis on vérifie sur des tournois jamais vus (test)."""
    rng = random.Random(rng_seed)
    best = params
    best_score = evaluate_params(best, train_seeds, workers)["roi"]
    history = [{"tour": 0, "roi_train": round(best_score, 3), "params": asdict(best)}]
    knobs = {"open_range": 0.04, "push_fold_bb": 2.0, "push_range_short": 0.05,
             "value_raise_eq": 0.04, "call_margin": 0.02, "bluff_freq": 0.03}
    for k in range(1, rounds + 1):
        name = rng.choice(list(knobs))
        delta = knobs[name] * rng.choice([-1, 1])
        cand = replace(best, **{name: round(max(0.0, getattr(best, name) + delta), 3)})
        sc = evaluate_params(cand, train_seeds, workers)["roi"]
        kept = sc > best_score
        if kept:
            best, best_score = cand, sc
        history.append({"tour": k, "essai": f"{name} {delta:+}", "roi_train": round(sc, 3),
                        "garde": kept})
    base_test = evaluate_params(params, test_seeds, workers)
    new_test = evaluate_params(best, test_seeds, workers)
    return {"params": asdict(best), "history": history,
            "test_avant": round(base_test["roi"], 3), "test_apres": round(new_test["roi"], 3),
            "n_test": len(test_seeds)}
