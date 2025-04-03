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

            # Calculate theoretical price
            theo = self.find_theo(order_depth)
            if theo is None:
                continue  # Skip if we can't calculate theo
            
            # Use volume-weighted market making
            self.volume_weighted_mm(product, theo, order_depth, orders, curr_position, position_limit)
            
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData

    def find_theo(self, order_depth) -> float:
        theo = None
        if order_depth.sell_orders and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate total volume at best bid/ask
            best_bid_vol = order_depth.buy_orders[best_bid]
            best_ask_vol = order_depth.sell_orders[best_ask]
            total_vol = best_bid_vol + best_ask_vol
            
            # Update price and volume history
            self.price_history.append(mid_price)
            self.volume_history.append(total_vol)
            
            # Simple moving average
            if len(self.price_history) >= self.WINDOW_SIZE:
                theo = np.mean(self.price_history[-self.WINDOW_SIZE:])
            else:
                theo = mid_price
        return theo

    def volume_weighted_mm(self, product, fair_price, order_depth, orders, position, pos_limit):
        """
        Volume-weighted strategy
        
        Parameters:
        product: Trading asset
        fair_price: Estimated fair value
        order_depth: Current order book
        orders: List to append orders to
        position: Current position
        position_limit: Max allowed position
        """

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return
            
        # Get best bid/ask and volumes
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = order_depth.sell_orders[best_ask]
        
        # Calculate volume imbalance and total volume
        total_vol = best_bid_vol + best_ask_vol
        volume_imbalance = 0 
        if total_vol > 0:
            volume_imbalance = (best_bid_vol - best_ask_vol) / total_vol
        
        # Dynamic spread based on volume imbalance
        base_spread = (best_ask - best_bid) * 0.8  # Start tighter than current spread
        dynamic_spread = base_spread * (1 + 0.5 * volume_imbalance)
        
        # Inventory adjustment - widen spread as position approaches limit
        inventory_adj = (position / pos_limit) * base_spread/2
        
        # Set our prices
        our_bid = fair_price - dynamic_spread/2 - inventory_adj
        our_ask = fair_price + dynamic_spread/2 - inventory_adj
        
        # Calculate order sizes based on volume and inventory
        max_bid_size = pos_limit - position
        max_ask_size = pos_limit + position
        
        # Volume-weighted size - take a fraction of available volume
        bid_size = min(int(best_bid_vol * 0.3), max_bid_size, self.DEFAULT_TRADE_SIZE)
        ask_size = min(int(best_ask_vol * 0.3), max_ask_size, self.DEFAULT_TRADE_SIZE)
        
        # Post bids if profitable
        if our_bid > best_bid and bid_size > 0:
            print(f"POST BID {bid_size}x {our_bid}")
            orders.append(Order(product, our_bid, bid_size))
        
        # Post asks if profitable
        if our_ask < best_ask and ask_size > 0:
            print(f"POST ASK {ask_size}x {our_ask}")
            orders.append(Order(product, our_ask, -ask_size))  # Negative for sell

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