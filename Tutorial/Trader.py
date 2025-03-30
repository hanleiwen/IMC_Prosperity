from datamodel import OrderDepth, UserId, TradingState, Order
import string
import pandas as pd
import numpy as np
import statistics as st
import math
from typing import List
import jsonpickle

class Trader:
    POS_LIMIT = 50
    DEFAULT_TRADE_SIZE = 5
    WINDOW_SIZE = 10  # For moving average
    price_history = []

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position_limit = self.POS_LIMIT
            curr_position = state.position.get(product, 0)
            theo = 10  ######

            self.market_make(product, theo, 1, order_depth, orders, curr_position, position_limit)
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData

    def find_theo(self, product, order_depth) -> float:
        theo = None
        if order_depth.sell_orders and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            # Update price history
            self.price_history.append(mid_price)
            
            # Simple moving average
            if len(self.price_history) >= self.WINDOW_SIZE:
                theo = np.mean(self.price_history[-self.WINDOW_SIZE:])
            else:
                theo = mid_price
        return theo

    def market_make(self, product, fair_val, spread, 
                     order_depth, orders, curr_pos, pos_limit):
        """Generic, symmetric spread market making function"""

        best_bid = 0
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())

        best_ask = float('inf')
        if order_depth.sell_orders:
            min(order_depth.sell_orders.keys())
        
        # Calculate our bid and ask prices
        our_bid = fair_val - spread
        our_ask = fair_val + spread
        
        # Bid logic
        if our_bid > best_bid:
            # Calculate how much we can buy without exceeding position limit
            buy_amount = min(pos_limit - curr_pos, self.DEFAULT_TRADE_SIZE)
            if buy_amount > 0:
                print(f"POST BID {buy_amount}x {our_bid}")
                orders.append(Order(product, our_bid, buy_amount))
        
        # Ask logic
        if our_ask < best_ask:
            # Calculate how much we can sell without exceeding short limit
            sell_amount = max(-pos_limit - curr_pos, -self.DEFAULT_TRADE_SIZE)  
            if sell_amount < 0:
                print(f"POST ASK {-sell_amount}x {our_ask}")
                orders.append(Order(product, our_ask, sell_amount))