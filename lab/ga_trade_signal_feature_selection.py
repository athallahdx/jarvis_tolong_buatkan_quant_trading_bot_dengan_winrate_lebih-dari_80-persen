import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "input":          "BTCUSDT_2024_6m_features_5min.csv",
    "train_end":      "2024-04-15",
    "val_end":        "2024-05-15",
    # GA
    "pop_size":       50,
    "generations":    100,
    "crossover_rate": 0.8,
    "mutation_rate":  0.02,
    "tournament_k":   3,
    "elite_n":        2,
    # fitness
    "min_trades":     10,
    "max_sharpe":     5.0,       # cap to prevent chasing lookahead artifacts
    "bars_per_year":  252 * 288, # 5-min bars, crypto 24/7
}

# returns / returns_5 / returns_10 removed — they cause lookahead
FEATURE_COLS = [
    # order flow
    "delta_ratio", "buy_sell_ratio", "cvd_slope_5", "cvd_slope_10",
    "cvd_zscore", "notional_buy_ratio", "notional_sell_ratio",
    "large_trade_imbalance", "large_trade_ratio", "trade_intensity",
    # price action (no raw returns)
    "hl_range", "bar_body", "upper_wick", "lower_wick",
    # moving averages
    "ema_cross_9_21", "ema_cross_21_50",
    # momentum
    "rsi_14", "rsi_7", "stoch_k", "stoch_d",
    # trend
    "adx", "adx_diff", "macd", "macd_signal", "macd_diff",
    # volatility
    "atr_ratio", "bb_width", "bb_pct",
    # volume
    "vol_zscore", "vol_ratio", "notional_zscore",
]
N_FEATURES = len(FEATURE_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(cfg):
    # ── LOAD ────────────────────────────────────────────────────────────────
    df = pd.read_csv(
        cfg["input"],
        parse_dates=["time"],
        index_col="time"
    ).sort_index()

    # ── CLEAN RETURNS (TARGET ONLY, NOT FEATURE) ───────────────────────────
    df["returns"] = df["close"].pct_change()

    # keep only required columns
    needed = FEATURE_COLS + ["open", "high", "low", "close", "returns"]
    df = df[needed]

    # ── SHIFT FEATURES TO PREVENT LOOKAHEAD ────────────────────────────────
    # signal at bar t only uses info available at t-1
    df[FEATURE_COLS] = df[FEATURE_COLS].shift(1)

    # remove NaNs after shift/pct_change
    df = df.dropna()

    # ── TIME SPLIT (NEVER SHUFFLE) ─────────────────────────────────────────
    train = df[df.index < cfg["train_end"]].copy()

    val = df[
        (df.index >= cfg["train_end"]) &
        (df.index < cfg["val_end"])
    ].copy()

    test = df[df.index >= cfg["val_end"]].copy()

    # ── FIT SCALER ONLY ON TRAIN (IMPORTANT) ───────────────────────────────
    scaler = RobustScaler()

    scaler.fit(train[FEATURE_COLS])

    # transform separately
    train[FEATURE_COLS] = scaler.transform(train[FEATURE_COLS])
    val[FEATURE_COLS]   = scaler.transform(val[FEATURE_COLS])
    test[FEATURE_COLS]  = scaler.transform(test[FEATURE_COLS])

    # ── CLIP TO REDUCE EXTREME OUTLIER IMPACT ──────────────────────────────
    train[FEATURE_COLS] = train[FEATURE_COLS].clip(-3, 3) / 3
    val[FEATURE_COLS]   = val[FEATURE_COLS].clip(-3, 3) / 3
    test[FEATURE_COLS]  = test[FEATURE_COLS].clip(-3, 3) / 3

    # ── INFO ───────────────────────────────────────────────────────────────
    print(f"Features      : {N_FEATURES}")

    print(
        f"Train         : {len(train):>6,} bars  "
        f"({train.index[0].date()} → {train.index[-1].date()})"
    )

    print(
        f"Validation    : {len(val):>6,} bars  "
        f"({val.index[0].date()} → {val.index[-1].date()})"
    )

    print(
        f"Test          : {len(test):>6,} bars  "
        f"({test.index[0].date()} → {test.index[-1].date()})"
    )

    return train, val, test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────
def random_individual():
    # ensure at least 3 features active at init
    mask = [0] * N_FEATURES
    for i in np.random.choice(N_FEATURES, size=max(3, N_FEATURES // 4), replace=False):
        mask[i] = 1
    return {
        "mask":    mask,
        "weights": [np.random.uniform(-1, 1) for _ in range(N_FEATURES)],
        "buy_th":  np.random.uniform(0.1, 0.7),
        "sell_th": np.random.uniform(-0.7, -0.1),
    }

def clone(ind):
    return {
        "mask":    ind["mask"].copy(),
        "weights": ind["weights"].copy(),
        "buy_th":  ind["buy_th"],
        "sell_th": ind["sell_th"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df, ind):
    feat_matrix  = df[FEATURE_COLS].values
    mask         = np.array(ind["mask"],    dtype=float)
    weights      = np.array(ind["weights"], dtype=float)
    active       = mask * weights
    n_active     = mask.sum()

    if n_active == 0:
        return pd.Series(0, index=df.index)

    scores  = feat_matrix @ active / n_active
    signals = np.where(scores >  ind["buy_th"],   1,
              np.where(scores <  ind["sell_th"],  -1, 0))
    return pd.Series(signals, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def backtest(df, signals):
    # ── KEY FIX: signal at bar t → trade at bar t+1 ───────────────────────
    # features already shifted in load_and_split, but shift signal again
    # as a double safety against lookahead
    shifted   = signals.shift(1).fillna(0)
    strat_ret = shifted * df["returns"]
    return strat_ret


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(strat_ret, bars_per_year):
    active        = strat_ret[strat_ret != 0]
    total_trades  = len(active)

    if total_trades == 0:
        return {"sharpe": -999, "total_return": -999,
                "max_dd": -999, "win_rate": 0, "trades": 0}

    mean   = strat_ret.mean()
    std    = strat_ret.std() + 1e-9
    sharpe = (mean / std) * np.sqrt(bars_per_year)

    cumret  = (1 + strat_ret).cumprod()
    max_dd  = (cumret / cumret.cummax() - 1).min()

    wins     = (active > 0).sum()
    win_rate = wins / total_trades

    total_return = cumret.iloc[-1] - 1

    return {
        "sharpe":       sharpe,
        "total_return": total_return,
        "max_dd":       max_dd,
        "win_rate":     win_rate,
        "trades":       total_trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FITNESS
# ─────────────────────────────────────────────────────────────────────────────

def fitness(ind, val_df, cfg):
    n_active = sum(ind["mask"])
    if n_active == 0:
        return -999.0

    signals  = generate_signals(val_df, ind)
    strat    = backtest(val_df, signals)
    metrics  = compute_metrics(strat, cfg["bars_per_year"])

    if metrics["trades"] < cfg["min_trades"]:
        return -999.0

    # ── KEY FIX: cap sharpe ────────────────────────────────────────────────
    sharpe     = min(metrics["sharpe"], cfg["max_sharpe"])

    dd_penalty = abs(metrics["max_dd"]) * 2
    parsimony  = (n_active / N_FEATURES) * 0.1

    return float(sharpe - dd_penalty - parsimony)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GA OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(population, scores, k):
    idx  = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax([scores[i] for i in idx])]
    return clone(population[best])

def crossover(p1, p2):
    point = np.random.randint(1, N_FEATURES)
    c1 = {
        "mask":    p1["mask"][:point]    + p2["mask"][point:],
        "weights": p1["weights"][:point] + p2["weights"][point:],
        "buy_th":  p1["buy_th"]  if np.random.rand() > 0.5 else p2["buy_th"],
        "sell_th": p1["sell_th"] if np.random.rand() > 0.5 else p2["sell_th"],
    }
    c2 = {
        "mask":    p2["mask"][:point]    + p1["mask"][point:],
        "weights": p2["weights"][:point] + p1["weights"][point:],
        "buy_th":  p2["buy_th"]  if np.random.rand() > 0.5 else p1["buy_th"],
        "sell_th": p2["sell_th"] if np.random.rand() > 0.5 else p1["sell_th"],
    }
    return c1, c2

def mutate(ind, rate):
    ind = clone(ind)
    for i in range(N_FEATURES):
        if np.random.rand() < rate:
            ind["mask"][i] ^= 1
        if np.random.rand() < rate:
            ind["weights"][i] += np.random.uniform(-0.2, 0.2)
            ind["weights"][i]  = float(np.clip(ind["weights"][i], -1, 1))
    if np.random.rand() < rate:
        ind["buy_th"]  = float(np.clip(
            ind["buy_th"]  + np.random.uniform(-0.05, 0.05), 0.05, 0.95))
        ind["sell_th"] = float(np.clip(
            ind["sell_th"] + np.random.uniform(-0.05, 0.05), -0.95, -0.05))
    # ensure at least 1 feature always active
    if sum(ind["mask"]) == 0:
        ind["mask"][np.random.randint(N_FEATURES)] = 1
    return ind


# ─────────────────────────────────────────────────────────────────────────────
# 8. GA MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(val_df, cfg):
    pop             = [random_individual() for _ in range(cfg["pop_size"])]
    history         = []
    best_ever       = None
    best_ever_score = -np.inf

    print(f"\n{'─'*62}")
    print(f"  GA  |  pop={cfg['pop_size']}  "
          f"gen={cfg['generations']}  "
          f"features={N_FEATURES}")
    print(f"{'─'*62}")

    for gen in range(cfg["generations"]):
        scores = [fitness(ind, val_df, cfg) for ind in pop]

        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score > best_ever_score:
            best_ever_score = best_score
            best_ever       = clone(pop[best_idx])

        valid_scores = [s for s in scores if s > -999]
        history.append({
            "generation": gen + 1,
            "best":       best_score,
            "mean":       float(np.mean(valid_scores)) if valid_scores else -999,
            "n_features": sum(pop[best_idx]["mask"]),
        })

        print(f"  Gen {gen+1:>3d}/{cfg['generations']}  "
              f"best={best_score:>6.4f}  "
              f"mean={history[-1]['mean']:>6.4f}  "
              f"feat={history[-1]['n_features']:>2d}/{N_FEATURES}")

        # elitism
        elite_idx = np.argsort(scores)[::-1][:cfg["elite_n"]]
        elites    = [clone(pop[i]) for i in elite_idx]

        # selection → crossover → mutation
        offspring = []
        while len(offspring) < cfg["pop_size"] - cfg["elite_n"]:
            p1 = tournament_select(pop, scores, cfg["tournament_k"])
            p2 = tournament_select(pop, scores, cfg["tournament_k"])
            if np.random.rand() < cfg["crossover_rate"]:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = clone(p1), clone(p2)
            offspring.append(mutate(c1, cfg["mutation_rate"]))
            offspring.append(mutate(c2, cfg["mutation_rate"]))

        pop = elites + offspring[:cfg["pop_size"] - cfg["elite_n"]]

    print(f"{'─'*62}")
    print(f"  DONE  |  best fitness = {best_ever_score:.4f}")
    print(f"{'─'*62}\n")
    return best_ever, pd.DataFrame(history)


# ─────────────────────────────────────────────────────────────────────────────
# 9. REPORT
# ─────────────────────────────────────────────────────────────────────────────

def interpret_sharpe(sharpe):
    if sharpe < 0:     return "❌ Losing"
    elif sharpe < 0.5: return "⚠️  Poor"
    elif sharpe < 1.0: return "🟡 Acceptable"
    elif sharpe < 2.0: return "✅ Good"
    elif sharpe < 3.0: return "✅ Very good"
    else:              return "🔍 Excellent — check overfit"

def report(label, df, ind, cfg):
    signals = generate_signals(df, ind)
    strat   = backtest(df, signals)
    m       = compute_metrics(strat, cfg["bars_per_year"])
    bh      = compute_metrics(df["returns"], cfg["bars_per_year"])

    print(f"\n{'═'*58}")
    print(f"  {label}")
    print(f"{'═'*58}")
    print(f"  {'Metric':<22} {'GA Strategy':>12}  {'Buy & Hold':>12}")
    print(f"  {'─'*48}")
    print(f"  {'Sharpe ratio':<22} {m['sharpe']:>12.4f}  {bh['sharpe']:>12.4f}")
    print(f"  {'Total return':<22} {m['total_return']:>11.2%}  {bh['total_return']:>11.2%}")
    print(f"  {'Max drawdown':<22} {m['max_dd']:>11.2%}  {bh['max_dd']:>11.2%}")
    print(f"  {'Win rate':<22} {m['win_rate']:>11.2%}  {'—':>12}")
    print(f"  {'Total trades':<22} {m['trades']:>12,}  {'—':>12}")
    print(f"  {'Sharpe verdict':<22} {interpret_sharpe(m['sharpe']):>12}")
    print(f"{'═'*58}")
    return m, strat, signals

def print_best_individual(ind):
    selected = [
        (FEATURE_COLS[i], ind["weights"][i])
        for i in range(N_FEATURES) if ind["mask"][i] == 1
    ]
    selected.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n── Best Individual ──────────────────────────────────")
    print(f"  Features : {len(selected)} / {N_FEATURES} selected")
    print(f"  Buy  th  : {ind['buy_th']:.4f}")
    print(f"  Sell th  : {ind['sell_th']:.4f}")
    print(f"\n  {'Feature':<30} {'Weight':>8}  {'Bar'}")
    print(f"  {'─'*55}")
    for feat, w in selected:
        bar = ("+" if w >= 0 else "-") * int(abs(w) * 12)
        print(f"  {feat:<30} {w:>+8.4f}  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_ga_history(history):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.spines[:].set_color("#333")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    gens = history["generation"]
    axes[0].plot(gens, history["best"], color="lime",    linewidth=1.5, label="Best")
    axes[0].plot(gens, history["mean"], color="#4da6ff", linewidth=1.0,
                 label="Mean", linestyle="--")
    axes[0].axhline(0, color="#555", linewidth=0.5)
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("GA Evolution", color="white")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(gens, history["n_features"], color="orange", linewidth=1.2)
    axes[1].set_ylabel("Active features")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylim(0, N_FEATURES + 1)
    axes[1].axhline(N_FEATURES // 2, color="#555", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig("ga_history.png", dpi=150, facecolor="#0f0f0f")
    plt.show()
    print("Saved → ga_history.png")

def plot_equity_curve(df, strat_ret, signals, title):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")
        ax.spines[:].set_color("#333")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")

    ga_eq = (1 + strat_ret).cumprod()
    bh_eq = (1 + df["returns"]).cumprod()
    axes[0].plot(df.index, ga_eq, color="lime",    linewidth=1.2, label="GA strategy")
    axes[0].plot(df.index, bh_eq, color="#4da6ff", linewidth=1.0,
                 label="Buy & hold", linestyle="--")
    axes[0].set_ylabel("Equity")
    axes[0].set_title(title, color="white")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(df.index, df["close"], color="white", linewidth=0.6)
    buy_idx  = df.index[signals ==  1]
    sell_idx = df.index[signals == -1]
    axes[1].scatter(buy_idx,  df.loc[buy_idx,  "close"],
                    marker="^", color="lime", s=12, zorder=5, label="Buy")
    axes[1].scatter(sell_idx, df.loc[sell_idx, "close"],
                    marker="v", color="red",  s=12, zorder=5, label="Sell")
    axes[1].set_ylabel("Price")
    axes[1].legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)

    drawdown = ga_eq / ga_eq.cummax() - 1
    axes[2].fill_between(df.index, drawdown, 0, color="red", alpha=0.4)
    axes[2].set_ylabel("Drawdown")

    plt.tight_layout()
    fname = "equity_" + title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150, facecolor="#0f0f0f")
    plt.show()
    print(f"Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def random_baseline(val_df, cfg, n_trials=30):
    best_score, best_ind = -np.inf, None
    for _ in range(n_trials):
        ind = random_individual()
        s   = fitness(ind, val_df, cfg)
        if s > best_score:
            best_score, best_ind = s, ind
    return best_ind

def equal_weight_baseline():
    w = 1 / N_FEATURES
    return {
        "mask":    [1] * N_FEATURES,
        "weights": [w] * N_FEATURES,
        "buy_th":  0.5,
        "sell_th": -0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # 1 — load
    print("\nLoading data...")
    train_df, val_df, test_df, scaler = load_and_split(CONFIG)

    # 2 — run GA
    best_ind, history = run_ga(val_df, CONFIG)

    # 3 — print best individual
    print_best_individual(best_ind)

    # 4 — ga evolution plot
    plot_ga_history(history)

    # 5 — validate
    val_m, val_strat, val_sig = report("VALIDATION SET", val_df, best_ind, CONFIG)
    plot_equity_curve(val_df, val_strat, val_sig, "Validation Set")

    # 6 — test (final, only touched once)
    test_m, test_strat, test_sig = report("TEST SET", test_df, best_ind, CONFIG)
    plot_equity_curve(test_df, test_strat, test_sig, "Test Set")

    # 7 — overfitting check
    ratio = val_m["sharpe"] / (abs(test_m["sharpe"]) + 1e-9)
    print(f"\n── Overfit Check ────────────────────────────────────")
    print(f"  Val  Sharpe : {val_m['sharpe']:.4f}  {interpret_sharpe(val_m['sharpe'])}")
    print(f"  Test Sharpe : {test_m['sharpe']:.4f}  {interpret_sharpe(test_m['sharpe'])}")
    print(f"  Val/Test    : {ratio:.2f}x  "
          f"{'⚠️  possible overfit' if ratio > 2.0 else '✅ generalizes ok'}")

    # 8 — baselines
    print("\nRunning baselines...")
    rand_ind  = random_baseline(val_df, CONFIG)
    equal_ind = equal_weight_baseline()
    rand_m,  _, _ = report("BASELINE: Random",       test_df, rand_ind,  CONFIG)
    equal_m, _, _ = report("BASELINE: Equal Weights", test_df, equal_ind, CONFIG)

    # 9 — final summary
    print(f"\n{'═'*62}")
    print(f"  FINAL COMPARISON — Test Set")
    print(f"{'═'*62}")
    print(f"  {'Method':<32} {'Sharpe':>7}  {'Return':>8}  {'MaxDD':>8}")
    print(f"  {'─'*56}")
    bh_m = compute_metrics(test_df["returns"], CONFIG["bars_per_year"])
    rows = [
        ("GA — order flow feature selection", test_m),
        ("Baseline: random selection",        rand_m),
        ("Baseline: equal weights",           equal_m),
        ("Buy & hold",                        bh_m),
    ]
    for name, m in rows:
        print(f"  {name:<32} {m['sharpe']:>7.4f}  "
              f"{m['total_return']:>7.2%}  {m['max_dd']:>7.2%}")
    print(f"{'═'*62}\n")

    # 10 — save results
    result = {
        "selected_features": [FEATURE_COLS[i] for i in range(N_FEATURES)
                               if best_ind["mask"][i] == 1],
        "n_features_selected": int(sum(best_ind["mask"])),
        "feature_mask":        best_ind["mask"],
        "weights":             best_ind["weights"],
        "buy_threshold":       best_ind["buy_th"],
        "sell_threshold":      best_ind["sell_th"],
        "val_sharpe":          val_m["sharpe"],
        "val_return":          val_m["total_return"],
        "val_max_dd":          val_m["max_dd"],
        "test_sharpe":         test_m["sharpe"],
        "test_return":         test_m["total_return"],
        "test_max_dd":         test_m["max_dd"],
        "test_win_rate":       test_m["win_rate"],
        "test_trades":         test_m["trades"],
        "overfit_ratio":       float(ratio),
    }
    with open("best_individual.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved → best_individual.json")