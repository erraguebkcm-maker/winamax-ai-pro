# WINAMAX AI PRO — robot d'analyse sportive et d'entraînement au poker

Prototype fonctionnel, testé. **Argent fictif uniquement. Aucune connexion à un compte de jeu, aucune mise automatique.**

---

## 1. Ce qui est réellement réalisable (Phase A)

| Demande | Faisable ? | Comment / limite |
|---|---|---|
| Robot poker qui joue seul 24 h/24 | **Oui, dans le simulateur interne** | Tournois Sit & Go simulés contre 5 styles d'adversaires virtuels. Interdit sur Winamax réel : les robots y sont bannis (comptes fermés et gains saisis lors d'affaires connues). |
| Analyse de tes parties | Oui, **après** les parties | À partir des historiques que tu exportes toi-même (phase D). Jamais pendant une partie. |
| Pronostics automatiques | Oui | Football d'abord (données gratuites de qualité). Tennis/NBA ensuite : données gratuites plus pauvres. |
| « Paris à valeur » | Techniquement oui, **rentabilité non garantie** | Les cotes des bookmakers sont déjà très précises. Sur les données de démo, le marché bat nos modèles : c'est le cas le plus fréquent dans la réalité aussi. |
| Blessures, compositions | Partiel | Aucune source gratuite fiable et autorisée. Le robot l'indique sur chaque fiche. |
| Placement des mises | **Non, volontairement** | Tu décides et tu mises toi-même. |
| Fonctionner téléphone éteint | Oui | GitHub Actions (gratuit) lance le robot chaque jour. |

## 2. Architecture

```
GitHub (gratuit)
 ├─ Robot quotidien (GitHub Actions, ~2 min/jour)
 │   ├─ Football : téléchargement football-data.co.uk → backtest chronologique
 │   │             → pronostics → journal horodaté des prédictions
 │   ├─ Poker    : 40 tournois simulés en parallèle → auto-amélioration prudente
 │   └─ Rapport  : report.json + rapport_AAAA-MM-JJ.md + tableau de bord HTML
 ├─ Tableau de bord (GitHub Pages) → ouvert sur Android, ajouté à l'écran d'accueil
 └─ Rapport quotidien → Telegram (facultatif)
```

```
winamax_ai_pro/
├─ robot.py                 ← le robot autonome (point d'entrée)
├─ config.py                ← tous les réglages (seuils, limites, championnats)
├─ dashboard.py + dashboard_template.html  ← interface mobile 7 écrans
├─ core/poker/
│   ├─ cards.py             ← évaluateur de mains
│   ├─ equity.py            ← équité Monte-Carlo, cotes du pot, force préflop
│   ├─ players.py           ← 5 adversaires virtuels + agent IA (modélisation adverse, push/fold)
│   ├─ tournament.py        ← moteur No Limit complet (blindes, tapis, pots annexes)
│   └─ trainer.py           ← exercices, comparaison reproductible, auto-amélioration
├─ core/sports/
│   ├─ data.py              ← données DÉMO + lecteur football-data.co.uk
│   ├─ models.py            ← Elo, Poisson bayésien, ensemble, marché, naïf
│   ├─ value.py             ← marge, espérance, incertitude, mise ¼ Kelly plafonnée
│   └─ engine.py            ← backtest walk-forward, Brier, log loss, bankroll fictive
├─ tests/test_all.py        ← 13 tests automatiques
├─ data/                    ← historique (journal des prédictions, état du robot poker)
├─ output/                  ← rapports et tableau de bord générés
└─ .github/workflows/robot.yml ← exécution quotidienne automatique
```

### Choix techniques
- **Python + numpy/pandas seulement** : rien à payer, rien à maintenir de compliqué.
- **Tableau de bord = page HTML statique** plutôt que Streamlit : pas de serveur à garder allumé, s'ouvre instantanément sur le téléphone, hébergement gratuit.
- **Modèles « en ligne »** : chaque match est prédit avant d'être vu, impossible d'utiliser le futur par accident (vérifié par un test).
- **Référence = le marché** : un modèle n'est déclaré « validé » que s'il bat les cotes ET si l'intervalle de confiance du rendement est au-dessus de 0.

## 3. Sources de données

| Source | Coût | Contenu | Utilisation |
|---|---|---|---|
| football-data.co.uk | Gratuit | Résultats + cotes historiques, 5 grands championnats, matchs à venir avec cotes | Utilisée par défaut en mode réel |
| football-data.org | Gratuit (12 compétitions, clé API) | Calendriers, résultats | Option future |
| API-Football, Sportmonks | Payant (offres d'essai) | Blessures, compositions | Seulement si le modèle prouve sa valeur |
| Tennis : fichiers de Jeff Sackmann (GitHub) | Gratuit | Résultats ATP/WTA | Phase suivante |

## 4. Installation (test sur ordinateur, facultatif)

```bash
pip install -r requirements.txt
python -m unittest -v      # 13 tests
python robot.py --quick    # journée rapide (8 tournois)
python robot.py            # journée complète
```
Ouvre ensuite `output/index.html` dans un navigateur.

## 5. Mise en service sur Android, sans ordinateur

1. Crée un compte gratuit sur **github.com** depuis le navigateur du téléphone.
2. Crée un dépôt nommé `winamax-ai-pro` et envoie-y les fichiers du projet (bouton *Add file → Upload files*, tu peux envoyer le dossier décompressé).
3. *Settings → Pages* : choisis **GitHub Actions** comme source.
4. *Actions* → « Robot WINAMAX AI PRO » → **Run workflow**. Après ~3 minutes, le lien du tableau de bord apparaît.
5. Ouvre le lien dans Chrome → menu ⋮ → **Ajouter à l'écran d'accueil**.
6. Rapport sur Telegram (facultatif) : crée un bot avec @BotFather, puis ajoute `TELEGRAM_TOKEN` et `TELEGRAM_CHAT_ID` dans *Settings → Secrets and variables → Actions*.

Passer aux vraies données : dans `config.py`, remplace `["DEMO"]` par `["F1"]` (Ligue 1) ou `["F1", "E0"]`.

⚠ Avec un compte GitHub gratuit, la page Pages est publique (adresse peu devinable, aucune donnée personnelle dedans : ton budget réel reste stocké sur ton téléphone uniquement).

## 6. Maintenance
- Chaque jour : lire le rapport. Rien d'autre à faire.
- Chaque mois : regarder l'onglet *Historique*. Si le modèle reste « non démontré » après 300+ prédictions réelles, ne pas miser sur ses sélections.
- Si l'exécution échoue : onglet *Actions* sur GitHub, l'étape en rouge indique la cause (souvent une source de données indisponible ce jour-là).
- Le journal `data/predictions_log.csv` est la preuve horodatée des prédictions faites avant les matchs.

## 7. Coûts de fonctionnement

| Poste | Coût |
|---|---|
| GitHub Actions (≈2 min/jour ; 2 000 min/mois gratuites en dépôt privé, illimité en public) | 0 € |
| GitHub Pages | 0 € |
| Données football-data.co.uk | 0 € |
| Telegram | 0 € |
| **Total** | **0 €/mois** |
| Option : données blessures/compos (API payante) | ~10 à 30 €/mois |

## 8. Règles respectées
- Pas de connexion au compte Winamax, pas de clic automatique, pas de lecture du logiciel de poker.
- Poker automatique uniquement dans le simulateur interne.
- Mise fictive au quart de Kelly, plafonnée à 2 % de la bankroll, stop-loss à −30 %, jamais de martingale (vérifié par un test).
- Aide si le jeu devient un problème : Joueurs Info Service 09 74 75 13 13.

## 9. Feuille de route
- **Fait** : phases A, B, C (démo), robot autonome, moteur poker, interface, tests, déploiement.
- **Ensuite** : passage aux vraies données Ligue 1 + suivi des prédictions réelles (phase C réelle), import des historiques de mains Winamax (phase D), tennis et NBA.
