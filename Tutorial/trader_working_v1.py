from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import math
import jsonpickle as jp

class Trader:
    def __init__(self):
        self.trader_data = []

    def fair(self, order_depth: OrderDepth, method = "mid_price", vol_filter = 0):
        best_ask = 0
        best_bid = 0
        mid_price = 0

        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
        elif method == "mid_price_with_vol_filter":
            if len(price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol_filter) == 0 or \
                len(price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol_filter) == 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol_filter])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol_filter])
        
        mid_price = (best_ask + best_bid) / 2
        return mid_price 
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, 
                             product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                allowed_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(allowed_quantity)))
                sell_order_volume += abs(allowed_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                allowed_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(allowed_quantity)))
                buy_order_volume += abs(allowed_quantity)
    
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, order_depth: OrderDepth, fair_value: int, width: int, position: int, 
                  position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        ''' ba = fair_value + width
        bb = fair_value - width ''' 

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position) # max amt to buy 
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity)) 
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position) # should be the max we can sell 
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, product, buy_order_volume, sell_order_volume, fair_value)

        ''' 
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, ba + width, buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("AMETHYSTS", bb - width, -sell_quantity))  # Sell order 
        '''

        return orders

    def run(self, state: TradingState):
        result = {}

        # params for rainforest resin
        rr_position_limit = 50
        rr_fair_value = 10000
        rr_width = 1

        # params for kelp
        kelp_position_limit = 50
        kelp_fair_value = 2025
        kelp_width = 1
        
        if "RAINFOREST_RESIN" in state.order_depths:
            rr_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            rr_orders = self.market_make("RAINFOREST_RESIN", state.order_depths["RAINFOREST_RESIN"], rr_fair_value, rr_width, rr_position, rr_position_limit)
            result["RAINFOREST_RESIN"] = rr_orders

        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            kelp_orders = self.market_make("KELP", state.order_depths["KELP"], kelp_fair_value, kelp_width, kelp_position, kelp_position_limit)
            result["KELP"] = kelp_orders

        conversions = 1

        self.trader_data = jp.encode( {} )

        return result, conversions, self.trader_data