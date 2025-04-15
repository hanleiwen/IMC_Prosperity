from datamodel import OrderDepth, TradingState, Order  #UserId
from typing import List, Dict, Any
import jsonpickle
import numpy as np
# import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    SYNTHETIC1 = "SYNTHETIC1"
    SPREAD1 = "SPREAD1"

    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SYNTHETIC2 = "SYNTHETIC2"
    SPREAD2 = "SPREAD2"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "fair_value": 0,
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 25,
        "reversion_beta": -0.229,
        "disregard_edge": 0,
        "join_edge": 1,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "fair_value": 0,
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "reversion_beta": -0.250,
        "disregard_edge": 0,
        "join_edge": 1,
        "default_edge": 1,
    },
    Product.SPREAD1: {
        "default_spread_mean": 379.50439988484239,
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 7,
        "target_position": 58,
    },
    Product.SPREAD2: {
        "default_spread_mean": 227.70263993090543,
        "default_spread_std": 45.647796,
        "spread_std_window": 27,
        "zscore_threshold": 4.2,
        "target_position": 34,
    }
}

PICNIC_BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}

PICNIC_BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFOREST_RESIN: 50, Product.KELP: 50, Product.SQUID_INK: 50}

    def kelp_fair(self, order_depth: OrderDepth, vol_filter = 0):
        if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol_filter]) == 0 or \
            len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol_filter]) == 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
        else:
            best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol_filter])
            best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol_filter])

        fair_price = (best_ask + best_bid) / 2
        return fair_price
    def squid_ink_fair(self, order_depth: OrderDepth, vol_filter = 0):
        if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol_filter]) == 0 or \
            len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol_filter]) == 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
        else:
            best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= vol_filter])
            best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= vol_filter])

        fair_price = (best_ask + best_bid) / 2
        return fair_price
    
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[int, int]:
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
    def get_synthetic_basket1_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = PICNIC_BASKET1_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = PICNIC_BASKET1_WEIGHTS[Product.JAMS]
        DJEMBES_PER_BASKET = PICNIC_BASKET1_WEIGHTS[Product.DJEMBES]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        djembes_best_bid = (
            max(order_depths[Product.DJEMBES].buy_orders.keys())
            if order_depths[Product.DJEMBES].buy_orders
            else 0
        )
        djembes_best_ask = (
            min(order_depths[Product.DJEMBES].sell_orders.keys())
            if order_depths[Product.DJEMBES].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
            + djembes_best_bid * DJEMBES_PER_BASKET
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
            + djembes_best_ask * DJEMBES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            djembes_bid_volume = (
                order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
                // DJEMBES_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume, djembes_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            djembes_ask_volume = (
                -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
                // DJEMBES_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume, djembes_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price  
    def convert_synthetic_basket1_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket1_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * PICNIC_BASKET1_WEIGHTS[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * PICNIC_BASKET1_WEIGHTS[Product.JAMS],
            )
            djembes_order = Order(
                Product.DJEMBES, djembes_price, quantity * PICNIC_BASKET1_WEIGHTS[Product.DJEMBES]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)
            component_orders[Product.DJEMBES].append(djembes_order)

        return component_orders
    def execute_spread1_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC1, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket1_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC1, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket1_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            return aggregate_orders
    def spread1_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD1]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD1]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD1]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    -self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread1_orders(
                    self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def get_synthetic_basket2_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CROISSANTS_PER_BASKET = PICNIC_BASKET2_WEIGHTS[Product.CROISSANTS]
        JAMS_PER_BASKET = PICNIC_BASKET2_WEIGHTS[Product.JAMS]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        croissants_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        croissants_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        jams_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        jams_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            croissants_best_bid * CROISSANTS_PER_BASKET
            + jams_best_bid * JAMS_PER_BASKET
        )
        implied_ask = (
            croissants_best_ask * CROISSANTS_PER_BASKET
            + jams_best_ask * JAMS_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            croissants_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
                // CROISSANTS_PER_BASKET
            )
            jams_bid_volume = (
                order_depths[Product.JAMS].buy_orders[jams_best_bid]
                // JAMS_PER_BASKET
            )
            implied_bid_volume = min(
                croissants_bid_volume, jams_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            croissants_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
                // CROISSANTS_PER_BASKET
            )
            jams_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[jams_best_ask]
                // JAMS_PER_BASKET
            )
            implied_ask_volume = min(
                croissants_ask_volume, jams_ask_volume 
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price    
    def convert_synthetic_basket2_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                croissants_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                jams_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                jams_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            croissants_order = Order(
                Product.CROISSANTS,
                croissants_price,
                quantity * PICNIC_BASKET2_WEIGHTS[Product.CROISSANTS],
            )
            jams_order = Order(
                Product.JAMS,
                jams_price,
                quantity * PICNIC_BASKET2_WEIGHTS[Product.JAMS],
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(croissants_order)
            component_orders[Product.JAMS].append(jams_order)

        return component_orders
    def execute_spread2_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
            return aggregate_orders
    def spread2_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD2]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD2]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread2_orders(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            rainforest_resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            rainforest_resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    rainforest_resin_position,
                )
            )
            rainforest_resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    rainforest_resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            rainforest_resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                rainforest_resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                rainforest_resin_take_orders + rainforest_resin_clear_orders + rainforest_resin_make_orders
            )
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )

            self.params[Product.KELP]["fair_value"] = self.kelp_fair(state.order_depths["KELP"], 20)

            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    self.params[Product.KELP]["fair_value"],
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    self.params[Product.KELP]["fair_value"],
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                self.params[Product.KELP]["fair_value"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )

            self.params[Product.SQUID_INK]["fair_value"] = self.squid_ink_fair(state.order_depths["SQUID_INK"], 10)

            squid_ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]["fair_value"],
                    self.params[Product.SQUID_INK]["take_width"],
                    squid_ink_position,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                )
            )
            squid_ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]["fair_value"],
                    self.params[Product.SQUID_INK]["clear_width"],
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            squid_ink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                self.params[Product.SQUID_INK]["fair_value"],
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            result[Product.SQUID_INK] = (
                squid_ink_take_orders + squid_ink_clear_orders + squid_ink_make_orders
            )

        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )

        spread1_orders = self.spread1_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD1],
        )

        if spread1_orders != None:
            result[Product.CROISSANTS] = spread1_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread1_orders[Product.JAMS]
            result[Product.DJEMBES] = spread1_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread1_orders[Product.PICNIC_BASKET1]


        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position2 = (
            state.position[Product.PICNIC_BASKET2]
            if Product.PICNIC_BASKET2 in state.position
            else 0
        )

        spread2_orders = self.spread2_orders(
            state.order_depths,
            Product.PICNIC_BASKET2,
            basket_position2,
            traderObject[Product.SPREAD2],
        )

        if spread2_orders != None:
            result[Product.CROISSANTS] = spread2_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread2_orders[Product.JAMS]
            result[Product.PICNIC_BASKET2] = spread2_orders[Product.PICNIC_BASKET2]

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData