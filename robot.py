"""ROBOT AUTONOME WINAMAX AI PRO — mode démonstration, argent fictif uniquement.

Une exécution = une « journée » de travail du robot :
  1. Football : récupère les données, backtest chronologique, pronostics des matchs à venir,
     enregistre les prédictions AVANT les matchs (journal), règle celles dont le résultat est connu.
  2. Poker : joue N tournois simulés en parallèle contre des adversaires virtuels,
     tente une amélioration prudente de sa stratégie, génère des exercices.
  3. Produit le rapport du jour (JSON + Markdown) et le tableau de bord mobile (HTML).
  4. Envoie le rapport sur Telegram si TELEGRAM_TOKEN et TELEGRAM_CHAT_ID sont définis.

Lancement :  python robot.py            (complet)
             python robot.py --quick    (rapide, pour tester)
Il ne se connecte à AUCUN compte de jeu et ne place AUCUNE mise.
"""
import argparse
import json
import os
import random
import urllib.parse
import urllib.request
from datetime import datetime, date
from pathlib import Path
from statistics import mean

import pandas as pd

from config import SETTINGS
from core.sports.data import make_demo_league, fetch_real, LEAGUES
from core.sports.engine import SportsEngine
from core.poker.players import AgentParams
from core.poker.trainer import evaluate_params, improve, make_exercise

ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "output"


def _load_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def run_sports(cfg, now):
    reports, alerts = [], []
    log_path = DATA / "predictions_log.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    new_rows = []
    for lg in cfg["leagues"]:
        if lg == "DEMO":
            hist, up = make_demo_league()
            name = "Championnat DÉMO (données synthétiques)"
        else:
            hist, up = fetch_real(lg)
            name = LEAGUES.get(lg, lg)
            if hist.empty:
                alerts.append(f"{name} : données indisponibles aujourd'hui, aucune analyse produite.")
                continue
        eng = SportsEngine(cfg)
        bt = eng.backtest(hist, warmup=min(380, len(hist) // 3))
        cards = eng.predict_upcoming(up, now)
        validated = bt["verdict"].startswith("Le modèle bat")
        for c in cards:
            c["validated"] = validated
            if c["selected"]:
                new_rows.append({"logged_at": now, "date": c["date"], "league": lg, "match": c["match"],
                                 "market": c["market"], "odds": c["odds"], "p_model": round(c["p_model"], 4),
                                 "ev": round(c["ev"], 4), "result": ""})
        n_sel = sum(c["selected"] for c in cards)
        if n_sel and not validated:
            alerts.append(f"{name} : {n_sel} sélection(s) au-dessus des seuils, mais le modèle n'a pas "
                          "prouvé son avantage → à observer seulement.")
        if bt["stopped"]:
            alerts.append(f"{name} : le stop-loss de la bankroll fictive s'est déclenché pendant le backtest.")
        if not cards:
            alerts.append(f"{name} : aucun match à venir avec cotes dans la source.")
        reports.append({"league": lg, "name": name, "source": "DEMO" if lg == "DEMO" else "football-data.co.uk",
                        "n_history": len(hist), "backtest": bt, "cards": cards, "validated": validated})
    # journal des prédictions faites AVANT les matchs (preuve horodatée)
    if new_rows:
        add = pd.DataFrame(new_rows)
        if not log.empty:
            key = ["date", "match", "market"]
            add = add[~add.set_index(key).index.isin(log.set_index(key).index)]
        log = pd.concat([log, add], ignore_index=True)
        log.to_csv(log_path, index=False)
    return reports, alerts, (0 if log.empty else len(log))


def run_poker(cfg, day_seed, quick):
    n = 8 if quick else cfg["poker_tournaments_per_day"]
    workers = cfg["poker_workers"]
    state_path = DATA / "poker_state.json"
    state = _load_json(state_path, {"params": AgentParams().__dict__, "days": []})
    params = AgentParams(**state["params"])
    seeds = [day_seed * 1000 + i for i in range(n)]
    res = evaluate_params(params, seeds, workers)
    by_style = {}
    for r in res["results"]:
        for st in r["opponents"]:
            by_style.setdefault(st, []).append(r["profit"])
    places = [r["place"] for r in res["results"]]
    imp = None
    if not quick and cfg["poker_improve_rounds"] > 0:
        train = [day_seed * 1000 + 500 + i for i in range(24)]
        test = [day_seed * 1000 + 800 + i for i in range(24)]
        imp = improve(params, train, test, rounds=cfg["poker_improve_rounds"], workers=workers,
                      rng_seed=day_seed)
        if imp["test_apres"] > imp["test_avant"]:  # adopté seulement si confirmé sur tournois jamais vus
            state["params"] = imp["params"]
    day = {"date": str(date.today()), "tournois": n, "roi": round(res["roi"], 3),
           "itm": round(res["itm"], 3), "victoires": round(res["wins"], 3)}
    state["days"] = (state["days"] + [day])[-60:]
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"today": day, "places": {str(p): places.count(p) for p in range(1, 7)},
            "by_style": {k: round(mean(v) / 10, 3) for k, v in by_style.items()},
            "hands": sum(r["hands"] for r in res["results"]),
            "params": state["params"], "improvement": imp, "history": state["days"],
            "exercises": [make_exercise(day_seed * 10 + i) for i in range(5)]}


def markdown_report(rep):
    L = [f"# Rapport du robot — {rep['generated']}", "", "Argent fictif uniquement. Aucune mise réelle.", ""]
    for a in rep["alerts"]:
        L.append(f"- ⚠ {a}")
    for s in rep["sports"]:
        bt = s["backtest"]
        L += ["", f"## {s['name']}", f"Verdict : {bt['verdict']}",
              f"Backtest : {bt['n_bets']} paris fictifs, ROI {bt['roi']:+.1%} "
              f"(IC95 {bt['roi_ci95'][0]:+.1%} à {bt['roi_ci95'][1]:+.1%}), bankroll fin {bt['bank_end']} €."]
        for c in [c for c in s["cards"] if c["selected"]][:5]:
            L.append(f"- {c['match']} · {c['market_name']} @ {c['odds']} · estimé {c['p_model']:.0%} "
                     f"vs marché {c['p_fair']:.0%} · EV {c['ev']:+.1%} · incertitude {c['unc_level']}")
        if not any(c["selected"] for c in s["cards"]):
            L.append("- Aucun pari ne présente d'avantage suffisamment fiable aujourd'hui.")
    p = rep["poker"]["today"]
    L += ["", "## Poker (simulation)", f"{p['tournois']} tournois, ROI {p['roi']:+.1%}, "
          f"places payées {p['itm']:.0%}, victoires {p['victoires']:.0%}."]
    return "\n".join(L)


def send_telegram(text):
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=20)
        return True
    except Exception as e:
        print("[telegram] échec :", e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    day_seed = int(date.today().strftime("%Y%m%d")) % 100000
    print("[robot] football…")
    sports, alerts, n_logged = run_sports(SETTINGS, now)
    print("[robot] poker…")
    poker = run_poker(SETTINGS, day_seed, args.quick)
    rep = {"generated": now, "settings": SETTINGS, "sports": sports, "alerts": alerts,
           "n_logged": n_logged, "poker": poker}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, default=str), encoding="utf-8")
    md = markdown_report(rep)
    (OUT / f"rapport_{date.today()}.md").write_text(md, encoding="utf-8")
    from dashboard import build
    build(rep, OUT)
    sent = send_telegram(md)
    print(md)
    print(f"[robot] terminé. Tableau de bord : output/index.html · Telegram : {'envoyé' if sent else 'non configuré'}")


if __name__ == "__main__":
    main()
