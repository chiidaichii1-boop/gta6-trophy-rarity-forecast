# GTA VI trophy rarity — week-one forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chiidaichii1-boop/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chiidaichii1-boop/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for [make.fun bounty **gta6-trophy-rarity-forecast-01**](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of the players Exophase tracks for the game who will have earned it at 2026-11-26T00:00:00Z, a week after launch, as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), from the house's reads of the game's trophy page up to the deadline. Scored by mean pinball loss on the logit scale against the rates the page shows at the target; a prize needs a loss strictly below the best house entry.

## The model

`main.py` (Python 3.13, standard library only) learns the forecast at run time from the case's own history. Everything is on the logit scale, against log-time since the release day began:

- **Median:** a least-squares fit of logit(rate) on ln(1 + hours) over the last 24 hours of reads, read at the target, with its slope shrunk by a quarter. Earn rates saturate, so a straight linear extrapolation over a 4.5-day horizon overshoots; the shrink is the hedge.
- **Spread:** the fit's own prediction error at the target combined with a 0.40-logit floor for the horizon and refresh uncertainty the fit cannot see (the answer is the rates as of Exophase's last refresh before the target).
- **Launch-night prior:** when a case's whole history is flat — no refresh seen yet — the forecast expects one: the median sits 0.70 logit units below the last rate and the spread floor rises to 0.90, since a refresh dilutes every rate as a flood of new, less-progressed players joins the tracked population.

## Validation

No local grade exists for this competition (the server grades every entry), so the model was checked locally instead:

- On the practice cases, scored against the launch-night refresh reconstructed from the competition's own account of it (3,868 players tracked and 4,276 trophies earned at 04:30Z, every rate but the 100% one lower): mean pinball loss about 0.47 across three reconstructions of the refresh's unrecorded per-trophy split, against the persistence baseline's 1.46.
- On two simulated final-like histories (one where rates saturate, one where they stay log-linear), the model scores 0.10 and 0.14, against persistence's 0.30 and 0.45.

## Running it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes `<id>.out` (one row per trophy, five quantiles each) to the output directory. Deterministic, standard library only, no network.
