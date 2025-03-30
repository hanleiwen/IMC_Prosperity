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

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            if product not in self.POS_LIMIT:
                continue
                
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position_limit = self.POS_LIMIT
            curr_position = state.position.get(product, 0)
            theo = 10  ######

            self.market_making(
                    product, theo, 1, order_depth, orders, curr_position, position_limit
                )
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < theo:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > theo:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData

    def market_making(self, product, acceptable_price, spread, 
                     order_depth, orders, current_position, position_limit):
        """Generic, symmetric spread market making function"""

        DEFAULT_SIZE = 5

        best_bid = 0
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())

        best_ask = float('inf')
        if order_depth.sell_orders:
            min(order_depth.sell_orders.keys())
        
        # Calculate our bid and ask prices
        our_bid = acceptable_price - spread
        our_ask = acceptable_price + spread
        
        # Bid logic
        if our_bid > best_bid:
            # Calculate how much we can buy without exceeding position limit
            buy_amount = min(position_limit - current_position, DEFAULT_SIZE)
            if buy_amount > 0:
                print(f"POST BID {buy_amount}x {our_bid}")
                orders.append(Order(product, our_bid, buy_amount))
        
        # Ask logic
        if our_ask < best_ask:
            # Calculate how much we can sell without exceeding short limit
            sell_amount = max(-position_limit - current_position, -DEFAULT_SIZE)  
            if sell_amount < 0:
                print(f"POST ASK {-sell_amount}x {our_ask}")
                orders.append(Order(product, our_ask, sell_amount))