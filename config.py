"""Réglages de WINAMAX AI PRO — modifiables sans toucher au reste du code."""

SETTINGS = {
    # Source des données : "DEMO" (synthétique, hors-ligne) ou un code championnat réel
    # football-data.co.uk : "F1" Ligue 1, "E0" Premier League, "SP1" Liga, "D1" Bundesliga, "I1" Serie A
    "leagues": ["DEMO"],

    # Filtres de sélection des paris à valeur
    "min_games": 8,            # matchs connus minimum par équipe
    "min_odds": 1.40,
    "max_odds": 5.00,
    "min_ev": 0.04,            # espérance minimale : +4 %
    "max_uncertainty": 0.08,
    "suspicious_edge": 0.12,   # au-delà, on soupçonne une info manquante plutôt qu'une « affaire »

    # Gestion du risque — argent FICTIF uniquement
    "bankroll_start": 1000.0,
    "kelly_fraction": 0.25,    # quart de Kelly
    "max_stake_pct": 0.02,     # jamais plus de 2 % de la bankroll par pari
    "stop_loss_pct": 0.30,     # arrêt automatique si la bankroll fictive perd 30 %
    "daily_max_bets": 5,

    # Limites de dépenses RÉELLES (suivi manuel, aucune mise automatique)
    "real_monthly_budget": 0.0,

    # Poker
    "poker_tournaments_per_day": 40,
    "poker_workers": 4,        # tournois simultanés
    "poker_improve_rounds": 3,
}
