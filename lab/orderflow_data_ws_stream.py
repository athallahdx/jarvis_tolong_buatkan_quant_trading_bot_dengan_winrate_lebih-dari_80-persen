import websocket, json

def on_message(ws, msg):
    data = json.loads(msg)
    price = data['p']
    qty   = data['q']
    side  = 'SELL' if data['m'] else 'BUY'
    print(f"{side} | price: {price} | qty: {qty}")

ws = websocket.WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@aggTrade",
    on_message=on_message
)
ws.run_forever()