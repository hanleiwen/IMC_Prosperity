from datamodel import OrderDepth, TradingState, Order  #UserId
from typing import List, Optional
# import string
import jsonpickle
# import math

class Product:
    def init(self, symb, f_v, j_e, def_e, p_l,
            spr = 1, c_w = 0, p_a = True, a_v = 15, 
            r_b = -0.229, d_e = 1, s_p_l = None):
        symbol: str = symb
        fair_value: float = f_v
        spread: float = spr
        clear_width: float = c_w
        prevent_adverse: bool = p_a
        adverse_voume: float = a_v
        reversion_beta: float = r_b
        disregard_edge: float = d_e
        join_edge: float = j_e
        default_edge: float = def_e
        position_limit: int = p_l
        soft_position_limit: Optional[int] = s_p_l

class Rainforest_Resin(Product):
    def __init__(self):
        super().__init__("RAINFOREST_RESIN", 10000, 2, 4, 50, 10, s_p_l = None)

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
                        orders.append(Order(product, best_ask, quantity))
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
                        orders.append(Order(product, best_bid, -1 * quantity))
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
            orders.append(Order(product, round(bid), buy_quantity))

        sell_quantity = product.position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

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
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
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
                position = state.position[prod_symbol]

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

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData