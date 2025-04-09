# For Copy, Paste and Editing

# Functions Not Needed / Not - Understood
 # def self.params[Product.KELP]["fair_value"](self, order_depth: OrderDepth, traderObject) -> float:
    #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_bid = max(order_depth.buy_orders.keys())
    #         filtered_ask = [
    #             price
    #             for price in order_depth.sell_orders.keys()
    #             if abs(order_depth.sell_orders[price])
    #             >= self.params[Product.KELP]["adverse_volume"]
    #         ]
    #         filtered_bid = [
    #             price
    #             for price in order_depth.buy_orders.keys()
    #             if abs(order_depth.buy_orders[price])
    #             >= self.params[Product.KELP]["adverse_volume"]
    #         ]
    #         mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
    #         mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
    #         if mm_ask == None or mm_bid == None:
    #             if traderObject.get("starfruit_last_price", None) == None:
    #                 mmmid_price = (best_ask + best_bid) / 2
    #             else:
    #                 mmmid_price = traderObject["starfruit_last_price"]
    #         else:
    #             mmmid_price = (mm_ask + mm_bid) / 2

    #         if traderObject.get("starfruit_last_price", None) != None:
    #             last_price = traderObject["starfruit_last_price"]
    #             last_returns = (mmmid_price - last_price) / last_price
    #             pred_returns = (
    #                 last_returns * self.params[Product.KELP]["reversion_beta"]
    #             )
    #             fair = mmmid_price + (mmmid_price * pred_returns)
    #         else:
    #             fair = mmmid_price
    #         traderObject["starfruit_last_price"] = mmmid_price
    #         return fair
    #     return None