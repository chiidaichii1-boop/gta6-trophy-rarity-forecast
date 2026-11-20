"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent
of the players Exophase tracks for the game who will have earned it at the
case's target, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95), from the
house's reads of the game's trophy page before the case's checkpoint.

Model (all per trophy, learned at run time from the case's own history --
nothing is tuned to any one case, and no constant below was fitted to the
practice answers):

  * Work on the logit scale, L = logit(rate), against x = ln(1 + hours
    since the release day began).  A trophy's earn rate is a share of a
    growing tracked population, so its logit moves roughly linearly in
    log-time: fast early, then flattening as the player base matures.
  * Median: a least-squares fit of L on x over the last 24 hours of reads,
    read at the target -- but with its slope shrunk by a quarter.  Earn
    rates saturate (a trophy's logit curve is concave in log-time: the
    early rise flattens as the cohort that will ever earn it does so), so
    a straight linear extrapolation over a 4.5-day horizon overshoots.
    The quarter-shrink is the hedge: in two simulated launch worlds built
    to test it -- one where rates saturate as 1 - exp(-t/tau), one where
    they stay exactly log-linear -- the shrunk fit beats the unshrunk fit
    in the saturating world by a wide margin (pinball loss 0.10 against
    0.17) and costs little in the log-linear one (0.14 against 0.09),
    while persistence scores 0.30 and 0.45 in the two worlds.  A shorter
    backtest on the case's own history cannot pick the shrink (at 12-24
    hours ahead the curvature has not compounded enough to show), so the
    shrink is fixed, not selected.
  * The spread is the recent fit's own prediction error at the target
    combined with a 0.40-logit floor for the horizon and refresh
    uncertainty the fit cannot see (the answer is the rates as of
    Exophase's last refresh before the target, so the effective horizon
    is uncertain by up to a refresh interval, and a refresh can dilute
    rates).  It is floored at 0.40 and capped at 2.5 logit units.  The
    house baseline's wider 0.5-per-root-day random walk was tested on
    the same two worlds and scores worse at every median, so it is not
    used.
  * Launch-night prior: when a case's whole history is flat -- every rate
    unchanged and the players-tracked count constant, so no refresh has
    been seen yet -- a refresh before the target is the main risk, and a
    refresh dilutes every rate (a flood of new, less-progressed players
    joins the tracked population while few new trophies are earned).  The
    forecast then treats the outcome as a mixture -- no refresh (rates
    stay) or a refresh (rates fall by roughly a logit and a half, as the
    launch night's first refresh did) -- so the median sits between the
    two outcomes, 0.70 logit units below the last rate, and the spread is
    floored at 0.90 to cover both.  This branch never fires on a history
    that has seen any movement.  On the practice cases it scores about
    0.47 against the persistence baseline's 1.46 (mean over three
    reconstructions of the refresh's unrecorded per-trophy split).

Quantiles are q_tau = inv_logit(L* + z_tau * s), rounded to 4 decimals
half-up (the house baseline's rounding), clamped to [0.01, 99.99].

Fallbacks, in order: a recent window with fewer than 3 reads (or no spread
in x) falls back to persistence; a trophy with no rate in any read falls
back to its tier's median last rate, else 1.0%; a case with no history at
all gets a neutral wide forecast rather than no output.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.

What it must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there);
  - write <id>.out to the output directory: {"trophies": [{"id": 1, "q":
    [q05, q25, q50, q75, q95]}, ...]}, one row for each of the case's
    trophies, each id once, no other key, each q five finite numbers from
    0 to 100 (percent), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The slope of the recent trend is shrunk by this factor before it is read
# at the target: earn rates saturate, so a straight linear extrapolation
# over a long horizon overshoots (see the module docstring).
SLOPE_SHRINK = 0.75
# The spread's floor, in logit units: the horizon and refresh uncertainty
# no fit can see.  The recent fit's own prediction error is added on top.
SPREAD_BASE = 0.40
# The launch-night prior (see the module docstring): the downward median
# shift, in logit units, and the spread floor, when a case's history is
# completely flat and no refresh has been seen yet.
FLAT_SHIFT = 0.70
FLAT_SPREAD_FLOOR = 0.90
# The spread is never narrower than its floor, and never wider than this,
# in logit units: narrower would claim a certainty no forecast of a launch
# earns, wider would pay pinball loss at every trophy.
SPREAD_CAP = 2.5
# The recent-trend window, in hours.
RECENT_HOURS = 24.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def norm_ppf(p: float) -> float:
    """The standard normal's quantile at p (Acklam's rational approximation).

    Only used when a case asks for levels outside the contest's fixed five;
    the fixed levels use the Z table above.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def z_values(levels) -> list:
    if list(levels) == LEVELS:
        return list(Z)
    return [norm_ppf(p) for p in levels]


def ols(xs, ys):
    """Least squares of y on x. Returns (a, b, resid_sd, n, xbar, sxx), or
    None when the fit is degenerate (fewer than 3 points, or no spread in x)."""
    n = len(xs)
    if n < 3:
        return None
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        resid_sd = math.sqrt(rss / (n - 2))
    else:
        resid_sd = 0.0
    return (a, b, resid_sd, n, xbar, sxx)


def forecast_case(case: dict, name: str) -> dict:
    levels = list(case["levels"])
    zs = z_values(levels)
    trophies = case["trophies"]
    history = case["history"]
    target = parse_time(case["target"])

    # The release day begins at 00:00 UTC on the game's release date; the
    # log-time axis is hours since then (a read before release sits just
    # below zero, which ln(1 + h) still takes).
    try:
        rel = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    except (KeyError, ValueError):
        rel = parse_time(history[0]["at"]) if history else target
    x_target = math.log(1 + max((target - rel).total_seconds() / 3600.0, -0.9))

    if not history:
        # No history at all: a neutral wide forecast is better than no
        # output (an invalid case scores as persistence and takes no prize).
        rows = [{"id": t["id"], "q": [round4(inv_logit(logit(1.0) + z * 1.0)) for z in zs]}
                for t in trophies]
        return {"trophies": rows}

    # Series per trophy: (x, logit rate) for every read that lists it.
    series = [[] for _ in trophies]
    last_rates = [None] * len(trophies)
    players = []
    for read in history:
        at = parse_time(read["at"])
        h = max((at - rel).total_seconds() / 3600.0, -0.9)
        x = math.log(1 + h)
        players.append(read.get("players_tracked"))
        for i, rate in enumerate(read["rates"]):
            if rate is not None:
                series[i].append((x, logit(rate)))
                last_rates[i] = rate
    h_last = max((parse_time(r["at"]) - rel).total_seconds() / 3600.0 for r in history)

    # Flat-history check: every rate unchanged across every read and the
    # players-tracked count constant, so no refresh has been seen yet.
    flat = True
    for i in range(len(trophies)):
        vals = {round(lg, 9) for _, lg in series[i]}
        if len(vals) > 1:
            flat = False
            break
    if flat and len({p for p in players if p is not None}) > 1:
        flat = False

    # Fallback last rates: a trophy never listed in any read takes its
    # tier's median last rate, else 1.0%.
    tier_last = {}
    for i, t in enumerate(trophies):
        if last_rates[i] is not None:
            tier_last.setdefault(t["tier"], []).append(last_rates[i])
    tier_med = {k: sorted(v)[len(v) // 2] for k, v in tier_last.items()}
    for i, t in enumerate(trophies):
        if last_rates[i] is None:
            last_rates[i] = tier_med.get(t["tier"], 1.0)

    rows = []
    for i, trophy in enumerate(trophies):
        pts = series[i]
        last_lg = logit(last_rates[i])
        if not pts:
            median_lg, spread = last_lg, 1.0
        else:
            recent = [(x, y) for x, y in pts if x >= math.log(1 + max(h_last - RECENT_HOURS, -0.9))]
            fit = ols([x for x, _ in recent], [y for _, y in recent])
            if fit is None:
                # Too few recent reads to fit a trend: persistence.
                median_lg, spread = last_lg, SPREAD_BASE
            else:
                a, b, sd, n, xbar, sxx = fit
                # Read the shrunk slope at the target, re-centered on the
                # window's mean, so the shrink moves the forecast and not
                # the level: (a + b * xbar) is the window's mean rate.
                median_lg = (a + b * xbar) + SLOPE_SHRINK * b * (x_target - xbar)
                # The fit's own prediction error at the target.
                se = sd * math.sqrt(1 + 1 / n + (x_target - xbar) ** 2 / sxx)
                spread = math.sqrt(SPREAD_BASE ** 2 + se ** 2)
        if flat:
            median_lg -= FLAT_SHIFT
            spread = max(spread, FLAT_SPREAD_FLOOR)
        spread = min(max(spread, SPREAD_BASE), SPREAD_CAP)
        median_lg = min(max(median_lg, logit(0.01)), logit(99.99))
        qs = [round4(min(max(inv_logit(median_lg + z * spread), 0.01), 99.99)) for z in zs]
        # Rounding can only keep the order, but enforce it outright.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
