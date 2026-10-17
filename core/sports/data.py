"""Sources de données football.

1) DONNÉES DE DÉMONSTRATION (synthétiques, générées par le programme) : servent à tester
   le moteur sans Internet. Elles sont marquées source="DEMO" partout et ne décrivent
   AUCUN vrai championnat.
2) DONNÉES RÉELLES : fichiers CSV gratuits de football-data.co.uk (résultats + cotes
   historiques, Ligue 1 = F1, Premier League = E0, Liga = SP1, Bundesliga = D1, Serie A = I1)
   et fixtures.csv (matchs à venir avec cotes). Téléchargées par le serveur.
"""
import io
import math
import random
import urllib.request
from datetime import date, timedelta
import numpy as np
import pandas as pd

LEAGUES = {"F1": "Ligue 1", "E0": "Premier League", "SP1": "Liga", "D1": "Bundesliga", "I1": "Serie A"}
BASE_URL = "https://www.football-data.co.uk"
COLS = ["date", "league", "home", "away", "hg", "ag", "odds_h", "odds_d", "odds_a",
        "odds_o25", "odds_u25", "source"]


def _poisson_1x2(lh, la, maxg=10):
    ph = [math.exp(-lh) * lh ** k / math.factorial(k) for k in range(maxg + 1)]
    pa = [math.exp(-la) * la ** k / math.factorial(k) for k in range(maxg + 1)]
    m = np.outer(ph, pa)
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), m


def make_demo_league(seed=42, n_teams=20, seasons=3, margin=0.06, book_noise=0.25):
    """Championnat fictif : forces cachées, buts tirés selon une loi de Poisson,
    cotes de bookmaker = vraie probabilité bruitée + marge. Renvoie (historique, à venir)."""
    rng = np.random.default_rng(seed)
    teams = [f"DEMO {chr(65 + i)}" for i in range(n_teams)]
    att = rng.normal(0, 0.25, n_teams)
    dfn = rng.normal(0, 0.20, n_teams)
    rows = []
    d0 = date(2023, 8, 12)

    def odds_for(i, j):
        lh = math.exp(0.28 + 0.22 + att[i] - dfn[j])
        la = math.exp(0.28 + att[j] - dfn[i])
        p_h, p_d, p_a, m = _poisson_1x2(lh, la)
        p_over = 1 - sum(m[a, b] for a in range(3) for b in range(3) if a + b <= 2)
        def price(ps):
            noisy = np.array(ps) * np.exp(rng.normal(0, book_noise / 3, len(ps)))
            noisy = noisy / noisy.sum() * (1 + margin)
            return [round(1 / q, 2) for q in noisy]
        return (lh, la), price([p_h, p_d, p_a]), price([p_over, 1 - p_over])

    upcoming = []
    for s in range(seasons + 1):
        att += rng.normal(0, 0.06, n_teams)   # les équipes évoluent d'une saison à l'autre
        dfn += rng.normal(0, 0.05, n_teams)
        pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
        rng.shuffle(pairs)
        per_day = n_teams // 2
        for k, (i, j) in enumerate(pairs):
            day = d0 + timedelta(days=365 * s + 3 * (k // per_day))
            (lh, la), o1x2, ou = odds_for(i, j)
            if s == seasons:  # saison en cours : 3 journées jouées, la 4e est « à venir »
                if k // per_day >= 4:
                    break
                if k // per_day == 3:
                    upcoming.append(dict(date=day, league="DEMO", home=teams[i], away=teams[j],
                                         hg=None, ag=None, odds_h=o1x2[0], odds_d=o1x2[1],
                                         odds_a=o1x2[2], odds_o25=ou[0], odds_u25=ou[1],
                                         source="DEMO"))
                    continue
            rows.append(dict(date=day, league="DEMO", home=teams[i], away=teams[j],
                             hg=int(rng.poisson(lh)), ag=int(rng.poisson(la)),
                             odds_h=o1x2[0], odds_d=o1x2[1], odds_a=o1x2[2],
                             odds_o25=ou[0], odds_u25=ou[1], source="DEMO"))
    hist = pd.DataFrame(rows, columns=COLS).sort_values("date").reset_index(drop=True)
    up = pd.DataFrame(upcoming, columns=COLS)
    return hist, up


def parse_football_data_csv(text: str, league: str) -> pd.DataFrame:
    """Convertit un CSV football-data.co.uk au format interne (cotes moyennes du marché)."""
    df = pd.read_csv(io.StringIO(text))
    pick = lambda *names: next((n for n in names if n in df.columns), None)
    ch, cd, ca = pick("AvgH", "B365H", "PSH"), pick("AvgD", "B365D", "PSD"), pick("AvgA", "B365A", "PSA")
    co, cu = pick("Avg>2.5", "B365>2.5"), pick("Avg<2.5", "B365<2.5")
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dt.date,
        "league": df["Div"] if "Div" in df.columns else league,
        "home": df["HomeTeam"], "away": df["AwayTeam"],
        "hg": df["FTHG"] if "FTHG" in df.columns else None,
        "ag": df["FTAG"] if "FTAG" in df.columns else None,
        "odds_h": df[ch] if ch else np.nan, "odds_d": df[cd] if cd else np.nan,
        "odds_a": df[ca] if ca else np.nan,
        "odds_o25": df[co] if co else np.nan, "odds_u25": df[cu] if cu else np.nan,
        "source": "football-data.co.uk",
    })
    return out.dropna(subset=["date", "home", "away"])


def fetch_real(league="F1", seasons=("2324", "2425", "2526", "2627")):
    """Télécharge l'historique + les matchs à venir. Fonctionne sur le serveur (GitHub Actions),
    pas dans un environnement sans Internet. Renvoie (historique, à venir)."""
    frames = []
    for s in seasons:
        url = f"{BASE_URL}/mmz4281/{s}/{league}.csv"
        try:
            txt = urllib.request.urlopen(url, timeout=30).read().decode("latin-1")
            frames.append(parse_football_data_csv(txt, league))
        except Exception as e:  # saison absente ou réseau
            print(f"[données] {url} indisponible : {e}")
    hist = pd.concat(frames).dropna(subset=["hg", "ag"]) if frames else pd.DataFrame(columns=COLS)
    try:
        txt = urllib.request.urlopen(f"{BASE_URL}/fixtures.csv", timeout=30).read().decode("latin-1")
        up = parse_football_data_csv(txt, league)
        up = up[up["league"] == league]
    except Exception as e:
        print(f"[données] fixtures indisponibles : {e}")
        up = pd.DataFrame(columns=COLS)
    hist["hg"] = hist["hg"].astype(int)
    hist["ag"] = hist["ag"].astype(int)
    return hist.sort_values("date").reset_index(drop=True), up
