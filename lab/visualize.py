import pandas as pd
import mplfinance as mpf

df = pd.read_csv("BTCUSDT_2024-01-01_2024-01-31_processed_10m/ALL_10min.csv", parse_dates=["time"], index_col="time")

# dark style
mc = mpf.make_marketcolors(
    up        = "lime",
    down      = "red",
    edge      = "inherit",
    wick      = "inherit",
    volume    = "inherit",
)

style = mpf.make_mpf_style(
    marketcolors = mc,
    facecolor    = "#0f0f0f",
    edgecolor    = "#333333",
    figcolor     = "#0f0f0f",
    gridcolor    = "#222222",
    gridstyle    = "--",
    y_on_right   = True,
    rc           = {
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
        "text.color":       "white",
    }
)

mpf.plot(df,
    type    = "candle",
    volume  = True,
    title   = "BTCUSDT 10min",
    style   = style,
    figsize = (16, 8),
)