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
    "pop_size":       100,
    "generations":    100,
    "crossover_rate": 0.8,
    "mutation_rate":  0.02,
    "tournament_k":   3,
    "elite_n":        2,
    # fitness
    "min_trades":     10,
    "max_sharpe":     5.0,
    "bars_per_year":  252 * 288,      # 5-min bars, crypto 24/7
    # TP/SL — dalam satuan ATR
    "tp_atr_mult":    2.0,            # TP  = entry ± (ATR × 2.0)   → RR 2:1
    "sl_atr_mult":    1.0,            # SL  = entry ∓ (ATR × 1.0)
    # biaya
    "fee":            0.0004,         # 0.04% taker fee
    "slippage":       0.0002,         # 0.02% slippage estimasi
}

FEATURE_COLS = [
    # order flow
    "delta_ratio", "buy_sell_ratio", "cvd_slope_5", "cvd_slope_10",
    "cvd_zscore", "notional_buy_ratio", "notional_sell_ratio",
    "large_trade_imbalance", "large_trade_ratio", "trade_intensity",
    # price action
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
    df = pd.read_csv(
        cfg["input"],
        parse_dates=["time"],
        index_col="time"
    ).sort_index()

    df["returns"] = df["close"].pct_change()

    # ── Hitung ATR raw (harga asli) sebelum scaling ──────────────────────────
    # atr_ratio = ATR / close  →  ATR_raw = atr_ratio × close
    # Kolom ini TIDAK di-scale, disimpan terpisah untuk keperluan TP/SL
    df["atr_raw"] = df["atr_ratio"] * df["close"]

    needed = FEATURE_COLS + ["open", "high", "low", "close", "returns", "atr_raw"]
    df = df[needed]

    # Shift fitur 1 bar agar tidak ada lookahead
    df[FEATURE_COLS] = df[FEATURE_COLS].shift(1)
    df["atr_raw"]    = df["atr_raw"].shift(1)   # ATR juga di-shift, pakai ATR bar sebelumnya
    df = df.dropna()

    train = df[df.index < cfg["train_end"]].copy()
    val   = df[(df.index >= cfg["train_end"]) & (df.index < cfg["val_end"])].copy()
    test  = df[df.index >= cfg["val_end"]].copy()

    # Scale hanya FEATURE_COLS, bukan atr_raw / OHLC
    scaler = RobustScaler()
    scaler.fit(train[FEATURE_COLS])

    for split in [train, val, test]:
        split[FEATURE_COLS] = scaler.transform(split[FEATURE_COLS])
        split[FEATURE_COLS] = split[FEATURE_COLS].clip(-3, 3) / 3

    print(f"Features      : {N_FEATURES}")
    print(f"Train         : {len(train):>6,} bars  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"Validation    : {len(val):>6,} bars  ({val.index[0].date()} → {val.index[-1].date()})")
    print(f"Test          : {len(test):>6,} bars  ({test.index[0].date()} → {test.index[-1].date()})")

    return train, val, test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def random_individual():
    n_active   = np.random.randint(3, max(4, N_FEATURES // 2))
    mask       = [0] * N_FEATURES
    active_idx = np.random.choice(N_FEATURES, size=n_active, replace=False)
    for i in active_idx:
        mask[i] = 1

    weights = []
    for i in range(N_FEATURES):
        weights.append(np.random.uniform(-1.0, 1.0) if mask[i] == 1 else 0.0)

    sell_th = np.random.uniform(-0.7, -0.05)
    buy_th  = np.random.uniform(max(0.05, sell_th + 0.1), 0.7)

    return {
        "mask":    mask,
        "weights": weights,
        "buy_th":  float(buy_th),
        "sell_th": float(sell_th),
    }


def clone(ind):
    return {
        "mask":    ind["mask"].copy(),
        "weights": ind["weights"].copy(),
        "buy_th":  float(ind["buy_th"]),
        "sell_th": float(ind["sell_th"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df, ind):
    feat_matrix    = df[FEATURE_COLS].values
    mask           = np.array(ind["mask"], dtype=float)
    weights        = np.array(ind["weights"], dtype=float)
    active_weights = mask * weights

    if np.sum(mask) == 0:
        return pd.Series(0, index=df.index)

    norm   = np.sum(np.abs(active_weights)) + 1e-9
    scores = (feat_matrix @ active_weights) / norm
    scores = np.clip(scores, -1, 1)

    signals = np.where(
        scores > ind["buy_th"],  1,
        np.where(scores < ind["sell_th"], -1, 0)
    )
    return pd.Series(signals, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BACKTEST DENGAN FIXED TP/SL ATR
# ─────────────────────────────────────────────────────────────────────────────
#
# Logika per bar:
#   • Tidak ada posisi  → cek sinyal bar ini → buka trade di close[i] + slippage
#   • Ada posisi        → cek apakah high/low bar ini mengenai TP atau SL
#       – hit TP        → exit di harga TP, catat profit
#       – hit SL        → exit di harga SL, catat loss
#       – hit keduanya  → gunakan SL (worst-case / konservatif)
#       – tidak kena    → lanjut, catat unrealized (mark-to-market)
#
# Return: (strat_ret Series, positions Series, trade_log list)
# ─────────────────────────────────────────────────────────────────────────────

def backtest_tpsl(df, signals, cfg):
    fee      = cfg["fee"]
    slippage = cfg["slippage"]
    tp_mult  = cfg["tp_atr_mult"]
    sl_mult  = cfg["sl_atr_mult"]

    closes   = df["close"].values
    highs    = df["high"].values
    lows     = df["low"].values
    atr_vals = df["atr_raw"].values
    sigs     = signals.values
    n        = len(df)

    pnl_arr  = np.zeros(n)
    pos_arr  = np.zeros(n)          # posisi yang aktif di setiap bar

    position    = 0                 # +1 long, -1 short, 0 flat
    entry_price = 0.0
    tp_price    = 0.0
    sl_price    = 0.0

    trade_log = []                  # untuk analisis lanjutan

    for i in range(1, n):
        # ── FLAT: tidak ada posisi aktif ─────────────────────────────────────
        if position == 0:
            sig = sigs[i]
            if sig != 0:
                atr = atr_vals[i]
                if atr <= 0 or np.isnan(atr):
                    continue        # skip bar jika ATR tidak valid

                # Eksekusi di close bar i + efek slippage
                entry_price = closes[i] * (1.0 + slippage * sig)
                position    = sig

                if position == 1:   # ── LONG ──
                    tp_price = entry_price + atr * tp_mult
                    sl_price = entry_price - atr * sl_mult
                else:               # ── SHORT ──
                    tp_price = entry_price - atr * tp_mult
                    sl_price = entry_price + atr * sl_mult

                # Biaya entry langsung dikurangi di bar pembukaan
                pnl_arr[i] -= (fee + slippage)
                pos_arr[i]  = position

                trade_log.append({
                    "entry_bar":   i,
                    "entry_time":  df.index[i],
                    "direction":   "long" if position == 1 else "short",
                    "entry_price": entry_price,
                    "tp_price":    tp_price,
                    "sl_price":    sl_price,
                    "atr_at_entry": atr,
                })

        # ── DALAM POSISI: cek TP / SL di bar berikutnya ──────────────────────
        else:
            high = highs[i]
            low  = lows[i]

            hit_tp = (position ==  1 and high >= tp_price) or \
                     (position == -1 and low  <= tp_price)
            hit_sl = (position ==  1 and low  <= sl_price) or \
                     (position == -1 and high >= sl_price)

            if hit_tp or hit_sl:
                # Kalau keduanya kena dalam 1 bar → pakai SL (worst case)
                if hit_sl:
                    exit_price = sl_price
                    exit_type  = "SL"
                else:
                    exit_price = tp_price
                    exit_type  = "TP"

                trade_pnl   = position * (exit_price - entry_price) / entry_price
                trade_pnl  -= (fee + slippage)     # biaya exit
                pnl_arr[i] += trade_pnl
                pos_arr[i]  = position              # masih terisi sampai bar exit

                # Update trade log entry terakhir
                if trade_log:
                    trade_log[-1].update({
                        "exit_bar":   i,
                        "exit_time":  df.index[i],
                        "exit_price": exit_price,
                        "exit_type":  exit_type,
                        "pnl":        trade_pnl,
                        "bars_held":  i - trade_log[-1]["entry_bar"],
                    })

                position = 0        # kembali flat

            else:
                # Mark-to-market: unrealized P&L bar ini
                pnl_arr[i] = position * (closes[i] - closes[i - 1]) / closes[i - 1]
                pos_arr[i] = position

    strat_ret = pd.Series(pnl_arr, index=df.index)
    positions = pd.Series(pos_arr, index=df.index)
    return strat_ret, positions, trade_log


# ─────────────────────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(strat_ret, positions, bars_per_year):
    strat_ret = strat_ret.fillna(0)
    equity    = (1 + strat_ret).cumprod()

    total_return = equity.iloc[-1] - 1

    mean = strat_ret.mean()
    std  = strat_ret.std()
    sharpe = (mean / std) * np.sqrt(bars_per_year) if std >= 1e-9 else -999.0

    rolling_max = equity.cummax()
    drawdown    = equity / rolling_max - 1
    max_dd      = drawdown.min()

    trade_changes = positions.diff().fillna(0)
    entries       = (trade_changes != 0).sum()
    total_trades  = int(entries / 2)

    active_returns = strat_ret[strat_ret != 0]
    win_rate       = float((active_returns > 0).mean()) if len(active_returns) > 0 else 0.0

    exposure = float((positions != 0).mean())

    return {
        "sharpe":       float(sharpe),
        "total_return": float(total_return),
        "max_dd":       float(max_dd),
        "win_rate":     float(win_rate),
        "trades":       int(total_trades),
        "exposure":     float(exposure),
    }


def compute_trade_stats(trade_log):
    """Statistik tambahan dari trade_log (trade yang sudah closed)."""
    closed = [t for t in trade_log if "exit_type" in t]
    if not closed:
        return {}

    pnls      = [t["pnl"] for t in closed]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    tp_hits   = [t for t in closed if t["exit_type"] == "TP"]
    sl_hits   = [t for t in closed if t["exit_type"] == "SL"]
    bars_held = [t["bars_held"] for t in closed]

    avg_win  = float(np.mean(wins))   if wins   else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0

    return {
        "closed_trades":  len(closed),
        "tp_hits":        len(tp_hits),
        "sl_hits":        len(sl_hits),
        "tp_rate":        len(tp_hits) / len(closed),
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "profit_factor":  abs(sum(wins) / sum(losses)) if losses else float("inf"),
        "avg_bars_held":  float(np.mean(bars_held)),
        "max_bars_held":  int(np.max(bars_held)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. FITNESS  (pakai backtest_tpsl)
# ─────────────────────────────────────────────────────────────────────────────

def fitness(ind, val_df, cfg):
    n_active = sum(ind["mask"])
    if n_active < 3:
        return -999.0

    signals              = generate_signals(val_df, ind)
    strat_ret, positions, _ = backtest_tpsl(val_df, signals, cfg)
    metrics              = compute_metrics(strat_ret, positions, cfg["bars_per_year"])

    trades = metrics["trades"]
    if trades < cfg["min_trades"]:
        return -999.0

    sharpe       = np.nan_to_num(metrics["sharpe"], nan=-999, posinf=cfg["max_sharpe"])
    sharpe       = np.clip(sharpe, -5, cfg["max_sharpe"])
    max_dd       = abs(np.nan_to_num(metrics["max_dd"], nan=1.0))
    total_return = np.nan_to_num(metrics.get("total_return", 0.0))
    exposure     = np.nan_to_num(metrics.get("exposure", 1.0))

    trade_factor       = min(np.sqrt(trades / 50), 1.0)
    dd_penalty         = (max_dd ** 0.5) * 2.0
    complexity_penalty = (n_active / N_FEATURES) * 0.15
    exposure_penalty   = 0.5 if exposure < 0.05 else 0.0
    return_bonus       = np.tanh(total_return * 5)

    score = (
        (sharpe * trade_factor)
        + return_bonus
        - dd_penalty
        - complexity_penalty
        - exposure_penalty
    )

    return float(score) if np.isfinite(score) else -999.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. GA OPERATORS
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(population, scores, k):
    idx  = np.random.choice(len(population), k, replace=False)
    best = idx[np.argmax([scores[i] for i in idx])]
    return clone(population[best])


def crossover(p1, p2):
    point = np.random.randint(1, N_FEATURES)

    def make_child(a, b):
        mask    = a["mask"][:point]    + b["mask"][point:]
        weights = a["weights"][:point] + b["weights"][point:]
        for i in range(N_FEATURES):
            if mask[i] == 0:
                weights[i] = 0.0

        sell_th = a["sell_th"] if np.random.rand() > 0.5 else b["sell_th"]
        buy_th  = a["buy_th"]  if np.random.rand() > 0.5 else b["buy_th"]

        if sell_th >= buy_th:
            sell_th = buy_th - 0.1

        return {
            "mask":    mask,
            "weights": weights,
            "buy_th":  float(np.clip(buy_th,  0.05,  0.95)),
            "sell_th": float(np.clip(sell_th, -0.95, -0.05)),
        }

    return make_child(p1, p2), make_child(p2, p1)


def mutate(ind, rate):
    ind = clone(ind)

    for i in range(N_FEATURES):
        if np.random.rand() < rate:
            ind["mask"][i] ^= 1
        if ind["mask"][i] == 1:
            if np.random.rand() < rate:
                ind["weights"][i] += np.random.uniform(-0.2, 0.2)
                ind["weights"][i]  = float(np.clip(ind["weights"][i], -1, 1))
        else:
            ind["weights"][i] = 0.0

    if np.random.rand() < rate:
        ind["buy_th"]  += np.random.uniform(-0.05, 0.05)
        ind["sell_th"] += np.random.uniform(-0.05, 0.05)

    ind["buy_th"]  = float(np.clip(ind["buy_th"],  0.05,  0.95))
    ind["sell_th"] = float(np.clip(ind["sell_th"], -0.95, -0.05))

    if ind["sell_th"] >= ind["buy_th"]:
        ind["sell_th"] = ind["buy_th"] - 0.1

    active = sum(ind["mask"])
    if active < 3:
        inactive_idx = [i for i in range(N_FEATURES) if ind["mask"][i] == 0]
        chosen = np.random.choice(inactive_idx, size=(3 - active), replace=False)
        for i in chosen:
            ind["mask"][i]    = 1
            ind["weights"][i] = np.random.uniform(-1, 1)

    return ind


# ─────────────────────────────────────────────────────────────────────────────
# 8. GA MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(val_df, cfg):
    pop = [random_individual() for _ in range(cfg["pop_size"])]

    history         = []
    best_ever       = None
    best_ever_score = -np.inf
    stagnation      = 0

    print(f"\n{'─'*72}")
    print(f" GA | pop={cfg['pop_size']} gen={cfg['generations']} features={N_FEATURES}")
    print(f" TP mult={cfg['tp_atr_mult']}×ATR  SL mult={cfg['sl_atr_mult']}×ATR  "
          f"RR={cfg['tp_atr_mult']/cfg['sl_atr_mult']:.1f}:1")
    print(f"{'─'*72}")

    for gen in range(cfg["generations"]):
        mutation_rate = cfg["mutation_rate"] * (0.995 ** gen)

        scores     = [fitness(ind, val_df, cfg) for ind in pop]
        best_idx   = int(np.argmax(scores))
        best_score = scores[best_idx]

        if best_score > best_ever_score:
            best_ever_score = best_score
            best_ever       = clone(pop[best_idx])
            stagnation      = 0
        else:
            stagnation += 1

        if stagnation >= 30:
            print("\nEarly stopping: no improvement.\n")
            break

        valid_scores  = [s for s in scores if s > -999]
        mean_score    = float(np.mean(valid_scores)) if valid_scores else -999
        active_counts = [sum(ind["mask"]) for ind in pop]
        avg_features  = np.mean(active_counts)

        history.append({
            "generation": gen + 1,
            "best":       best_score,
            "mean":       mean_score,
            "avg_feat":   avg_features,
        })

        print(
            f" Gen {gen+1:>3d}/{cfg['generations']} "
            f"best={best_score:>7.4f} "
            f"mean={mean_score:>7.4f} "
            f"avg_feat={avg_features:>5.2f} "
            f"mut={mutation_rate:.4f}"
        )

        elite_idx   = np.argsort(scores)[::-1][:cfg["elite_n"]]
        elites      = [clone(pop[i]) for i in elite_idx]

        offspring   = []
        target_size = cfg["pop_size"] - cfg["elite_n"]

        while len(offspring) < target_size:
            p1 = tournament_select(pop, scores, cfg["tournament_k"])
            p2 = tournament_select(pop, scores, cfg["tournament_k"])

            if np.random.rand() < cfg["crossover_rate"]:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = clone(p1), clone(p2)

            offspring.append(mutate(c1, mutation_rate))
            if len(offspring) < target_size:
                offspring.append(mutate(c2, mutation_rate))

        n_immigrants = max(1, cfg["pop_size"] // 20)
        for _ in range(n_immigrants):
            replace_idx            = np.random.randint(len(offspring))
            offspring[replace_idx] = random_individual()

        pop = elites + offspring

    print(f"{'─'*72}")
    print(f" DONE | best fitness = {best_ever_score:.4f}")
    print(f"{'─'*72}\n")

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
    signals                      = generate_signals(df, ind)
    strat_ret, positions, t_log  = backtest_tpsl(df, signals, cfg)
    m                            = compute_metrics(strat_ret, positions, cfg["bars_per_year"])
    ts                           = compute_trade_stats(t_log)

    # Buy & hold
    bh_pos = pd.Series(1.0, index=df.index)
    bh     = compute_metrics(df["returns"], bh_pos, cfg["bars_per_year"])

    rr = cfg["tp_atr_mult"] / cfg["sl_atr_mult"]

    print(f"\n{'═'*62}")
    print(f"  {label}")
    print(f"  TP={cfg['tp_atr_mult']}×ATR  SL={cfg['sl_atr_mult']}×ATR  RR={rr:.1f}:1")
    print(f"{'═'*62}")
    print(f"  {'Metric':<28} {'GA Strategy':>12}  {'Buy & Hold':>10}")
    print(f"  {'─'*54}")
    print(f"  {'Sharpe ratio':<28} {m['sharpe']:>12.4f}  {bh['sharpe']:>10.4f}")
    print(f"  {'Total return':<28} {m['total_return']:>11.2%}  {bh['total_return']:>9.2%}")
    print(f"  {'Max drawdown':<28} {m['max_dd']:>11.2%}  {bh['max_dd']:>9.2%}")
    print(f"  {'Win rate':<28} {m['win_rate']:>11.2%}  {'—':>10}")
    print(f"  {'Total trades':<28} {m['trades']:>12,}  {'—':>10}")
    print(f"  {'Sharpe verdict':<28} {interpret_sharpe(m['sharpe'])}")

    if ts:
        print(f"\n  ── Trade Detail (TP/SL) ─────────────────────────────")
        print(f"  {'Closed trades':<28} {ts['closed_trades']:>12,}")
        print(f"  {'TP hits':<28} {ts['tp_hits']:>12,}  ({ts['tp_rate']:>6.1%})")
        print(f"  {'SL hits':<28} {ts['sl_hits']:>12,}  ({1-ts['tp_rate']:>6.1%})")
        print(f"  {'Avg win (per trade)':<28} {ts['avg_win']:>11.4%}")
        print(f"  {'Avg loss (per trade)':<28} {ts['avg_loss']:>11.4%}")
        print(f"  {'Profit factor':<28} {ts['profit_factor']:>12.3f}")
        print(f"  {'Avg bars held':<28} {ts['avg_bars_held']:>12.1f}")
        print(f"  {'Max bars held':<28} {ts['max_bars_held']:>12,}")

    print(f"{'═'*62}")
    return m, strat_ret, positions, signals, t_log


def print_best_individual(ind, cfg):
    selected = [
        (FEATURE_COLS[i], ind["weights"][i])
        for i in range(N_FEATURES) if ind["mask"][i] == 1
    ]
    selected.sort(key=lambda x: abs(x[1]), reverse=True)

    rr = cfg["tp_atr_mult"] / cfg["sl_atr_mult"]

    print(f"\n── Best Individual ──────────────────────────────────────")
    print(f"  Features : {len(selected)} / {N_FEATURES} selected")
    print(f"  Buy  th  : {ind['buy_th']:.4f}")
    print(f"  Sell th  : {ind['sell_th']:.4f}")
    print(f"  TP mult  : {cfg['tp_atr_mult']}×ATR")
    print(f"  SL mult  : {cfg['sl_atr_mult']}×ATR")
    print(f"  RR ratio : {rr:.1f}:1")
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
    axes[0].set_title("GA Evolution (TP/SL ATR Backtest)", color="white")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(gens, history["avg_feat"], color="orange", linewidth=1.2)
    axes[1].set_ylabel("Active features")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylim(0, N_FEATURES + 1)
    axes[1].axhline(N_FEATURES // 2, color="#555", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    plt.savefig("ga_history.png", dpi=150, facecolor="#0f0f0f")
    plt.show()
    print("Saved → ga_history.png")


def plot_equity_curve(df, strat_ret, positions, signals, title, cfg):
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

    axes[0].plot(df.index, ga_eq, color="lime",    linewidth=1.2, label="GA strategy (TP/SL)")
    axes[0].plot(df.index, bh_eq, color="#4da6ff", linewidth=1.0,
                 label="Buy & hold", linestyle="--")
    axes[0].set_ylabel("Equity")
    rr = cfg["tp_atr_mult"] / cfg["sl_atr_mult"]
    axes[0].set_title(f"{title}  |  TP={cfg['tp_atr_mult']}×ATR  SL={cfg['sl_atr_mult']}×ATR  RR={rr:.1f}:1",
                      color="white")
    axes[0].legend(facecolor="#1a1a1a", labelcolor="white")

    axes[1].plot(df.index, df["close"], color="white", linewidth=0.6)
    buy_idx  = df.index[signals ==  1]
    sell_idx = df.index[signals == -1]
    axes[1].scatter(buy_idx,  df.loc[buy_idx,  "close"],
                    marker="^", color="lime", s=12, zorder=5, label="Buy signal")
    axes[1].scatter(sell_idx, df.loc[sell_idx, "close"],
                    marker="v", color="red",  s=12, zorder=5, label="Sell signal")
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

    print("\nLoading data...")
    train_df, val_df, test_df, scaler = load_and_split(CONFIG)

    # ── GA ───────────────────────────────────────────────────────────────────
    best_ind, history = run_ga(val_df, CONFIG)
    print_best_individual(best_ind, CONFIG)
    plot_ga_history(history)

    # ── Evaluasi Validation ──────────────────────────────────────────────────
    val_m, val_strat, val_pos, val_sig, val_log = report(
        "VALIDATION SET", val_df, best_ind, CONFIG
    )
    plot_equity_curve(val_df, val_strat, val_pos, val_sig, "Validation Set", CONFIG)

    # ── Evaluasi Test ────────────────────────────────────────────────────────
    test_m, test_strat, test_pos, test_sig, test_log = report(
        "TEST SET", test_df, best_ind, CONFIG
    )
    plot_equity_curve(test_df, test_strat, test_pos, test_sig, "Test Set", CONFIG)

    # ── Overfit Check ────────────────────────────────────────────────────────
    ratio = val_m["sharpe"] / (abs(test_m["sharpe"]) + 1e-9)
    print(f"\n── Overfit Check ────────────────────────────────────────")
    print(f"  Val  Sharpe : {val_m['sharpe']:.4f}  {interpret_sharpe(val_m['sharpe'])}")
    print(f"  Test Sharpe : {test_m['sharpe']:.4f}  {interpret_sharpe(test_m['sharpe'])}")
    print(f"  Val/Test    : {ratio:.2f}x  "
          f"{'⚠️  possible overfit' if ratio > 2.0 else '✅ generalizes ok'}")

    # ── Baselines ────────────────────────────────────────────────────────────
    print("\nRunning baselines...")
    rand_ind  = random_baseline(val_df, CONFIG)
    equal_ind = equal_weight_baseline()

    rand_m,  _, _, _, _ = report("BASELINE: Random",        test_df, rand_ind,  CONFIG)
    equal_m, _, _, _, _ = report("BASELINE: Equal Weights", test_df, equal_ind, CONFIG)

    bh_pos = pd.Series(1.0, index=test_df.index)
    bh_m   = compute_metrics(test_df["returns"], bh_pos, CONFIG["bars_per_year"])

    # ── Final Comparison ─────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print(f"  FINAL COMPARISON — Test Set")
    print(f"  (TP={CONFIG['tp_atr_mult']}×ATR  SL={CONFIG['sl_atr_mult']}×ATR  "
          f"RR={CONFIG['tp_atr_mult']/CONFIG['sl_atr_mult']:.1f}:1)")
    print(f"{'═'*66}")
    print(f"  {'Method':<32} {'Sharpe':>7}  {'Return':>8}  {'MaxDD':>8}")
    print(f"  {'─'*60}")
    rows = [
        ("GA — order flow + TP/SL ATR", test_m),
        ("Baseline: random selection",  rand_m),
        ("Baseline: equal weights",     equal_m),
        ("Buy & hold",                  bh_m),
    ]
    for name, m in rows:
        print(f"  {name:<32} {m['sharpe']:>7.4f}  "
              f"{m['total_return']:>7.2%}  {m['max_dd']:>7.2%}")
    print(f"{'═'*66}\n")

    # ── Simpan Hasil ─────────────────────────────────────────────────────────
    test_ts = compute_trade_stats(test_log)
    result  = {
        "selected_features":   [FEATURE_COLS[i] for i in range(N_FEATURES)
                                 if best_ind["mask"][i] == 1],
        "n_features_selected": int(sum(best_ind["mask"])),
        "feature_mask":        best_ind["mask"],
        "weights":             best_ind["weights"],
        "buy_threshold":       best_ind["buy_th"],
        "sell_threshold":      best_ind["sell_th"],
        "tp_atr_mult":         CONFIG["tp_atr_mult"],
        "sl_atr_mult":         CONFIG["sl_atr_mult"],
        "rr_ratio":            CONFIG["tp_atr_mult"] / CONFIG["sl_atr_mult"],
        # validation
        "val_sharpe":          val_m["sharpe"],
        "val_return":          val_m["total_return"],
        "val_max_dd":          val_m["max_dd"],
        # test
        "test_sharpe":         test_m["sharpe"],
        "test_return":         test_m["total_return"],
        "test_max_dd":         test_m["max_dd"],
        "test_win_rate":       test_m["win_rate"],
        "test_trades":         test_m["trades"],
        "test_tp_rate":        test_ts.get("tp_rate", None),
        "test_profit_factor":  test_ts.get("profit_factor", None),
        "test_avg_bars_held":  test_ts.get("avg_bars_held", None),
        "overfit_ratio":       float(ratio),
    }
    with open("best_individual.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved → best_individual.json")