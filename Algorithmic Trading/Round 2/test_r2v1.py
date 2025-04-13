from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState  # UserId
from typing import List, Optional, Any
# import string
import jsonpickle
import json
# import math

class Product:
    def __init__(self, symb, f_v, j_e, def_e, p_l,
            spr = 1, c_w = 0, p_a = True, a_v = 15, 
            r_b = -0.229, d_e = 1, s_p_l = None):
        self.symbol: str = symb
        self.fair_value: float = f_v
        self.spread: float = spr
        self.clear_width: float = c_w
        self.prevent_adverse: bool = p_a
        self.adverse_volume: float = a_v
        self.reversion_beta: float = r_b
        self.disregard_edge: float = d_e
        self.join_edge: float = j_e
        self.default_edge: float = def_e
        self.position_limit: int = p_l
        self.soft_position_limit: Optional[int] = s_p_l

class Rainforest_Resin(Product):
    def __init__(self):
        super().__init__("RAINFOREST_RESIN", 10000, 2, 4, 50, 10, s_p_l = 10)

class Kelp(Product):
    def __init__(self):
        super().__init__("KELP", 0, 0, 1, 50)

class Squid_Ink(Product):
    def __init__(self):
        super().__init__("SQUID_INK", 0, 0, 1, 50)

class Croissant(Product):
    def __init__(self):
        super().__init__("CROISSANT", 0, 0, 1, 250)

class Jam(Product):
    def __init__(self):
        super().__init__("JAM", 0, 0, 1, 350)

class Djembe(Product):
    def __init__(self):
        super().__init__("DJEMBE", 0, 0, 1, 60)

class Basket(Product):
    def __init__(self, symb, f_v, j_e, def_e, p_l):
        super().__init__(symb, f_v, j_e, def_e, p_l)
        self.components: dict[str, int] = {}

class Picnic_Basket1(Basket):
    def __init__(self):
        super().__init__("PICNIC_BASKET1", 0, 0, 1, 60)
        self.components = {
            "CROISSANT": 6,
            "JAM": 3,
            "DJEMBE": 1
        }
        
class Picnic_Basket2(Basket):
    def __init__(self):
        super().__init__("PICNIC_BASKET2", 0, 0, 1, 100)
        self.components = {
            "CROISSANT": 4,
            "JAM": 2,
        }

class Trader:
    def __init__(self, products = None):
        if products is None:
            self.products: dict[str, Product] = {
                "RAINFOREST_RESIN": Rainforest_Resin(),
                "KELP": Kelp(),
                "SQUID_INK": Squid_Ink(),
                "CROISSANT": Croissant(),
                "JAM": Jam(),
                "DJEMBE": Djembe(),
                "PICNIC_BASKET1": Picnic_Basket1(),
                "PICNIC_BASKET2": Picnic_Basket2()
            }
        else:
            self.products = products

    def fair(self, order_depth: OrderDepth, method: str = "mid_price", vol_filter: int = 0):
        
        fair_price = None

        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

            if best_ask is not None and best_bid is not None:
                fair_price = (best_ask + best_bid) / 2
        
        elif method == "mid_price_with_vol_filter":
            filtered_asks = [price for price in order_depth.sell_orders.keys() 
                           if abs(order_depth.sell_orders[price]) > vol_filter] if order_depth.sell_orders else []
            filtered_bids = [price for price in order_depth.buy_orders.keys() 
                           if abs(order_depth.buy_orders[price]) > vol_filter] if order_depth.buy_orders else []

            if filtered_asks and filtered_bids:
                best_ask = min(filtered_asks)
                best_bid = max(filtered_bids)
            else:
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

            if best_ask is not None and best_bid is not None:
                fair_price = (best_ask + best_bid) / 2

        return fair_price

    def take_best_orders(
        self,
        product: Product,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        # prevent_adverse: bool = False,
        # adverse_volume: int = 0,
    ) -> tuple[int, int]:
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not product.prevent_adverse or abs(best_ask_amount) <= product.adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, 
                                   product.position_limit - position) 
                    if quantity > 0:
                        orders.append(Order(product.symbol, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not product.prevent_adverse or abs(best_bid_amount) <= product.adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, 
                                   product.position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product.symbol, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: Product,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        buy_quantity = product.position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product.symbol, round(bid), buy_quantity))

        sell_quantity = product.position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product.symbol, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: Product,
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

        buy_quantity = product.position_limit - (position + buy_order_volume)
        sell_quantity = product.position_limit + (position - sell_order_volume)

        if position_after_take > 0 and order_depth.buy_orders:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product.symbol, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and order_depth.sell_orders:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product.symbol, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        # prevent_adverse: bool = False,
        # adverse_volume: int = 0,
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
            # prevent_adverse,
            # adverse_volume,
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
        product: Product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        manage_position: bool = False,
        # disregard_edge: float, 
        # join_edge: float,  # join trades within this edge
        # default_edge: float,  # default edge to request if there are no levels to penny or join
        soft_position_limit: int = 0, # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + product.disregard_edge
        ] if order_depth.sell_orders else []
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - product.disregard_edge
        ] if order_depth.buy_orders else []

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + product.default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= product.join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - product.default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= product.join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            spl = product.soft_position_limit if product.soft_position_limit is not None else soft_position_limit

            if position > spl:
                ask -= 1
            elif position < -1 * spl:
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

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        for prod_symbol, order_depth in state.order_depths.items():
            if prod_symbol in self.products:
                product = self.products[prod_symbol]
                position = state.position.get("KELP", 0)

                if prod_symbol in ["KELP", "SQUID_INK"]:
                    product.fair_value = self.fair(order_depth, "mid_price_with_vol_filter", 20)

                take_orders, buy_vol, sell_vol = self.take_orders(
                    product,
                    order_depth,
                    product.fair_value,
                    product.spread // 2,
                    position
                )

                clear_orders, buy_vol, sell_vol = self.clear_orders(
                    product,
                    order_depth,
                    product.fair_value,
                    product.clear_width,
                    position,
                    buy_vol,
                    sell_vol
                )

                make_orders, _, _ = self.make_orders(
                    product,
                    order_depth,
                    product.fair_value,
                    position,
                    buy_vol,
                    sell_vol,
                    manage_position = (prod_symbol == "RAINFOREST_RESIN")
                )

                result[prod_symbol] = take_orders + clear_orders + make_orders

        conversions = 0
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()