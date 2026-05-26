import requests
from datetime import datetime

def get_trades(symbol, start, end):
    url = "https://api.binance.com/api/v3/aggTrades"
    trades = []

    while start < end:
        r = requests.get(url, params={
            "symbol":    symbol,
            "startTime": start,
            "endTime":   end,
            "limit":     1000
        })
        data = r.json()
        if not data:
            break

        trades.extend(data)
        start = data[-1]['T'] + 1  # next page from last timestamp

    return trades

# define your time range
start = int(datetime(2025, 5, 20, 9, 0).timestamp() * 1000)   # 2025-05-20 09:00
end   = int(datetime(2025, 5, 20, 10, 0).timestamp() * 1000)  # 2025-05-20 10:00

trades = get_trades("BTCUSDT", start, end)

for t in trades[:5]:
    print({
        "side":  "SELL" if t["m"] else "BUY",
        "price": t["p"],
        "qty":   t["q"],
        "time":  datetime.fromtimestamp(t["T"] / 1000)
    })

print(f"Total trades: {len(trades)}")