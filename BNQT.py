import pandas as pd
from binance.client import Client
from datetime import datetime as dt
import requests
from datetime import timezone
import numpy as np

class QT():
    def __init__(self):
        self.client = Client(api_key='8OrsgihqUsnxE5OgU8x80mdHVBb8GT9mCmwfJhoAEznzUlKHezy5XWiCsOQlTr3w',
                             api_secret='f0ZGsQBT09hk67oByjsZ2jKLKsHsZnQv0OL4ZLMXQfC9HxQo9zCMyyundWrj6Vv7')
        self.client.API_URL = 'https://testnet.binance.vision/api'
        self.symbol = 'ETHUSDT'
        self.order = {}

    def get_asset(self, currence):
        client = self.client
        asset = client.get_asset_balance(asset=currence)
        print(asset)

    def get_price(self, symbol):
        client = self.client
        tickprice = client.get_symbol_ticker(symbol=symbol)
        print(tickprice['price'])

    def order_buy(self, fund, symbol):
        client = self.client

        tickprice = client.get_symbol_ticker(symbol=symbol)
        buyprice = float(tickprice['price'])
        buy_quantity = round(fund / buyprice)
        print("buy_mount：", buy_quantity)
        info = client.get_symbol_info(symbol)
        if buy_quantity >= float(info['filters'][2]['minQty']):
            buy_limit = client.order_limit_buy(symbol=symbol, quantity=buy_quantity, price=buyprice)
            dt_str = dt.now().strftime("%Y%m%d%H%M%S")
            self.order[dt_str] = {"symbol": symbol,
                                  "orderId": buy_limit['orderId'],
                                  "sell": buy_quantity,
                                  "dollars": buy_quantity * buyprice,
                                  "type": "buy"}
        else:
            print("lastBuy:", info['filters'][2]['minQty'])
            print("Your order too less to buy")

    def order_sell(self, symbol, type, SPT=0, fund=None):
        # SPT = sell persentage
        client = self.client

        tickprice = client.get_symbol_ticker(symbol=symbol)
        sellprice = round(float(tickprice['price']), 6)
        asset = client.get_asset_balance(asset=type)
        print("asset", asset['free'])
        if fund is not None:
            sell_quantity = round(fund / sellprice, 3)
        else:
            sell_quantity = round(float(asset['free']) * SPT, 3)
        print("sell_mount：", sell_quantity)
        print("dollars", sell_quantity * sellprice)

        sell_limit = client.order_limit_sell(symbol=symbol, quantity=sell_quantity, price=100000)
        dt_str = dt.now().strftime("%Y%m%d%H%M%S")
        self.order[dt_str] = {"symbol": symbol,
                              "orderId": sell_limit['orderId'],
                              "sell": sell_quantity,
                              "dollars": sell_quantity * sellprice,
                              "type": "sell"}
    # 這裡在未來要有一個重新掛單的功能，目前有點懶得用
    def order_cancel(self, dt_str):
        client = self.client
        cancel = client.cancel_order(symbol=self.order[dt_str]["symbol"], orderId=self.order[dt_str]['orderId'])
        print(cancel)
        # 這邊準備要加一個，如果成功取消才有 pop
        self.order.pop(dt_str, None)



    def get_binance_data_request_(self,ticker, interval='1h', limit='all', start='2019-01-01 00:00:00'):
        """
        interval: str tick interval - 4h/1h/1d ...
        """
        Dstart = pd.to_datetime(start)
        columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
                   'taker_base_vol', 'taker_quote_vol', 'ignore']
        start = int(dt(Dstart.year,Dstart.month,Dstart.day,Dstart.hour,Dstart.minute,tzinfo=timezone.utc).timestamp() * 1000)
        url = f'https://www.binance.com/api/v3/klines?symbol={ticker}&interval={interval}&limit={limit}&startTime={start}'
        data = pd.DataFrame(requests.get(url).json(), columns=columns, dtype=float)
        data.index = [pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S') for x in data.open_time]
        usecols = ['open', 'high', 'low', 'close', 'volume', 'qav', 'num_trades', 'taker_base_vol',
                   'taker_quote_vol']
        data = data[usecols]
        print(data)
        # print(self.client.get_historical_trades(symbol='ETHUSDT'))
        klines = self.client.get_historical_klines('ETHUSDT', interval, "1 Jan,2021")



if __name__ == "__main__":
    qt = QT()

    qt.get_binance_data_request_('ETHUSDT', '1h')
    #qt.get_price(symbol=qt.symbol)
    # qt.get_asset(currence="ETH")
    # qt.order_sell(qt.symbol, type="ETH", fund=1500)
