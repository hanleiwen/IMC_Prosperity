from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation  # UserId
from typing import List, Optional, Any
# import string
import jsonpickle
import json
import numpy as np
from math import log, sqrt, erf, exp, pi

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
        self.price_history: list[float] = []

class Rainforest_Resin(Product):
    def __init__(self):
        super().__init__("RAINFOREST_RESIN", 10000, 2, 4, 50, 10, s_p_l = 10)

class Kelp(Product):
    def __init__(self):
        super().__init__("KELP", 2020, 0, 1, 50)

class Squid_Ink(Product):
    def __init__(self):
        super().__init__("SQUID_INK", 2000, 2, 4, 50, spr=3, 
                         c_w=3, p_a=True, a_v=10)

class Croissant(Product):
    def __init__(self):
        super().__init__("CROISSANTS", 4320, 2, 4, 250, spr=2, 
                         c_w=2, p_a=True, a_v=20)

class Jam(Product):
    def __init__(self):
        super().__init__("JAMS", 6600, 2, 4, 350, spr=2, 
                         c_w=2, p_a=True, a_v=20)

class Djembe(Product):
    def __init__(self):
        super().__init__("DJEMBES", 13400, 2, 4, 60, spr=3, 
                         c_w=3, p_a=True, a_v=10, s_p_l=15)

class Basket(Product):
    def __init__(self, symb, f_v, j_e, def_e, p_l):
        super().__init__(symb, f_v, j_e, def_e, p_l)
        self.components: dict[str, int] = {}

class Picnic_Basket1(Basket):
    def __init__(self):
        super().__init__("PICNIC_BASKET1", 58000, 0, 1, 60)
        self.components = {
            "CROISSANT": 6,
            "JAM": 3,
            "DJEMBE": 1
        }
        
class Picnic_Basket2(Basket):
    def __init__(self):
        super().__init__("PICNIC_BASKET2", 30000, 0, 1, 100)
        self.components = {
            "CROISSANT": 4,
            "JAM": 2,
        }

class Volcanic_Rock(Product):
    def __init__(self):
        super().__init__("VOLCANIC_ROCK", 10200, 2, 4, 300, spr=3, c_w=3, p_a=True, a_v=30, s_p_l=50)
        
class Voucher(Product):
    def __init__(self, symb: str, s_p: int, p_l: int = 200):
        super().__init__(symb, s_p, 1, 2, p_l, spr=3, c_w=3, p_a=True, a_v=15)
        self.strike_price = s_p
        self._last_iv = 0.3
        self.implied_vol_history = []
        self.moneyness_history = []
        self.base_iv_history = []
        
        # Time parameters (1M ticks/day, 7 days to 2 days expiry)
        self.ticks_per_day = 1000000
        self.total_days = 7
        self.min_days = 2

    def black_scholes_iv(self, S: float, V: float, T: float) -> float:
        """Proper erf() implementation of IV calculation"""
        intrinsic = max(S - self.strike_price, 0)
        if T <= 1e-6 or V <= intrinsic + 1e-4:
            return 0.0
        
        ln_SK = log(S/self.strike_price)
        sqrt_T = sqrt(T)
        vol = getattr(self, '_last_iv', 0.2)
        
        for _ in range(10):  # Reduced iterations
            d1 = (ln_SK + (vol**2/2)*T) / (vol*sqrt_T)
            d2 = d1 - vol*sqrt_T
            
            # Correct erf() to CDF conversion:
            bs_price = S * (0.5 + 0.5*erf(d1/sqrt(2))) - \
                    self.strike_price * (0.5 + 0.5*erf(d2/sqrt(2)))
            
            # Proper vega calculation
            vega = S * sqrt_T * exp(-d1**2/2) / sqrt(2*pi)
            
            price_diff = bs_price - V
            if abs(price_diff) < 1e-4 or vega < 1e-10:
                break
                
            vol -= price_diff / max(vega, 1e-4)
            vol = max(0.05, min(2.0, vol))
        
        self._last_iv = vol
        return vol

    def update_iv_curve(self, underlying_price: float, timestamp: int) -> float:
        """Updates both raw IVs and base IV (ATM volatility)"""
        days_passed = timestamp / self.ticks_per_day
        days_left = max(self.min_days, self.total_days - days_passed)
        T = days_left / 252  # Annualized
        
        if not self.price_history:
            return None
            
        current_price = self.price_history[-1]
        iv = self.black_scholes_iv(underlying_price, current_price, T)
        
        # Store raw IV for all strikes
        self.implied_vol_history.append(iv)
        
        # Calculate and store moneyness
        m_t = log(self.strike_price/underlying_price)/sqrt(T)
        self.moneyness_history.append(m_t)
        
        # Fit volatility smile and extract base IV (ATM)
        if len(self.moneyness_history) % 1000 == 0 and len(self.moneyness_history) > 10:
            try:
                coeffs = np.polyfit(self.moneyness_history[-10:], 
                                  self.implied_vol_history[-10:], 2)
                self.iv_smile = np.poly1d(coeffs)
                base_iv = float(self.iv_smile(0))  # IV at moneyness=0 (ATM)
                self.base_iv_history.append(base_iv)
            except:
                pass
        
        return iv

    def get_iv_signal(self) -> int:
        """Uses base_iv_history for more stable signals"""
        if len(self.base_iv_history) < 5:
            return 0  # Neutral if insufficient data
            
        current_iv = self.base_iv_history[-1]
        iv_mean = np.mean(self.base_iv_history[-5:])
        iv_std = np.std(self.base_iv_history[-5:])
        
        # Dynamic signal thresholds
        if iv_std < 1e-4:  # Flat market
            return 0
            
        z_score = (current_iv - iv_mean) / iv_std
        
        if z_score > 1.5:
            return -1  # Strong sell (overpriced)
        elif z_score > 0.5:
            return -2  # Mild sell
        elif z_score < -1.5:
            return 1   # Strong buy (underpriced)
        elif z_score < -0.5:
            return 2   # Mild buy
        return 0

class Volcanic_Rock_Voucher_9500(Voucher):
    def __init__(self):
        super().__init__("VOLCANIC_ROCK_VOUCHER_9500", 9500)

class Volcanic_Rock_Voucher_9750(Voucher):
    def __init__(self):
        super().__init__("VOLCANIC_ROCK_VOUCHER_9750", 9750)

class Volcanic_Rock_Voucher_10000(Voucher):
    def __init__(self):
        super().__init__("VOLCANIC_ROCK_VOUCHER_10000", 10000)

class Volcanic_Rock_Voucher_10250(Voucher):
    def __init__(self):
        super().__init__("VOLCANIC_ROCK_VOUCHER_10250", 10250)

class Volcanic_Rock_Voucher_10500(Voucher):
    def __init__(self):
        super().__init__("VOLCANIC_ROCK_VOUCHER_10500", 10500)

class Magnificent_Macarons(Product):
    def __init__(self):
        super().__init__("MAGNIFICENT MACARONS", 640, 2, 4, 75, spr=3, 
            c_w=2, p_a=True, a_v=10)
        self.CSI = 0.5  # (to be calibrated)
        self.conversion_limit = 10
        self.storage_cost = 0.1
    
    def update_fair_value(self, conv_obs: ConversionObservation) -> None:
        """Update fair value based on sunlight and sugar prices"""
        if conv_obs:
            sunlight_factor = 1.2 if conv_obs.sunlightIndex < self.CSI else 1.0
            sugar_impact = (conv_obs.sugarPrice - 50) * 2  # 2 seashells per sugar unit
            self.fair_value = 640 * sunlight_factor + sugar_impact

    def calculate_conversion(self, conv_obs: ConversionObservation, position: int) -> int:
        """Determine optimal conversion quantity considering all fees"""
        if not conv_obs or position == 0:
            return 0
            
        if position > 0:  # Long position - consider selling to Pristine
            effective_price = conv_obs.bidPrice - conv_obs.transportFees - conv_obs.exportTariff
            if effective_price > self.fair_value:
                return -min(position, self.conversion_limit)
        else:  # Short position - consider buying from Pristine
            effective_price = conv_obs.askPrice + conv_obs.transportFees + conv_obs.importTariff
            if effective_price < self.fair_value:
                return min(abs(position), self.conversion_limit)
        return 0

class Trader:
    def __init__(self, products = None):
        if products is None:
            self.products: dict[str, Product] = {
                "RAINFOREST_RESIN": Rainforest_Resin(),
                "KELP": Kelp(),
                "SQUID_INK": Squid_Ink(),
                "CROISSANTS": Croissant(),
                "JAMS": Jam(),
                "DJEMBES": Djembe(),
                "PICNIC_BASKET1": Picnic_Basket1(),
                "PICNIC_BASKET2": Picnic_Basket2()
                # "VOLCANIC_ROCK": Volcanic_Rock(),
                # "VOLCANIC_ROCK_VOUCHER_9500": Volcanic_Rock_Voucher_9500(),
                # "VOLCANIC_ROCK_VOUCHER_9750": Volcanic_Rock_Voucher_9750(),
                # "VOLCANIC_ROCK_VOUCHER_10000": Volcanic_Rock_Voucher_10000(),
                # "VOLCANIC_ROCK_VOUCHER_10250": Volcanic_Rock_Voucher_10250(),
                # "VOLCANIC_ROCK_VOUCHER_10500": Volcanic_Rock_Voucher_10500(),
                # "MAGNIFICENT_MACARONS": Magnificent_Macarons()
            }
        else:
            self.products = products

    def get_liquidity_score(self, prod_symbol: str, order_depth: OrderDepth) -> float:
        """Score 0-1 where >0.7 is high liquidity"""
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total_vol = bid_vol + ask_vol
        return min(total_vol / max(300, self.products[prod_symbol].position_limit * 4), 1.0)

    def mid_price_fair(self, order_depth: OrderDepth):
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        if best_ask is not None and best_bid is not None:
            return (best_ask + best_bid) / 2
        
        return None
    
    def vol_filter_fair(self, order_depth: OrderDepth, vol_filter: int):
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
            return (best_ask + best_bid) / 2
        
        return None
    
    def moving_average_fair(self, prod_symbol: str, order_depth: OrderDepth, window_size: int):
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        self.products[prod_symbol].price_history.append(mid_price)
        if len(self.products[prod_symbol].price_history) > window_size:
            self.products[prod_symbol].price_history.pop(0)

        return sum(self.products[prod_symbol].price_history) / len(self.products[prod_symbol].price_history)

    def fair(self, product_symbol: str, state: TradingState, order_depth: OrderDepth):    
        if product_symbol in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            fair_price = 0
            for p, q in self.products[product_symbol].components.items():
                if p in state.order_depths:
                    fair_price += self.fair(p, state, state.order_depths[p]) * q
    
            return fair_price 

        liquidity = self.get_liquidity_score(product_symbol, order_depth)

        window_bounds = {
            'high': (3, 7),
            'medium': (6, 12),
            'low': (10, 20)
        }
        w_z = int(window_bounds["low"][0] + (window_bounds["low"][1] - window_bounds["low"][0]) * (1 - liquidity))
        v_f = self.products[product_symbol].position_limit * 0.1

        if liquidity > 0.7:
            w_z = int(window_bounds["high"][0] + (window_bounds["high"][1] - window_bounds["high"][0]) * (1 - liquidity))
            v_f += self.products[product_symbol].position_limit * 0.2
        elif liquidity > 0.3:
            w_z = int(window_bounds["medium"][0] + (window_bounds["medium"][1] - window_bounds["medium"][0]) * (1 - liquidity))
            v_f += self.products[product_symbol].position_limit * 0.1

        mp_fair = self.mid_price_fair(order_depth)
        vf_fair = self.vol_filter_fair(order_depth, v_f)
        ma_fair = self.moving_average_fair(product_symbol, order_depth, w_z)

        return 0.1 * mp_fair + 0.45 * vf_fair + 0.45 * ma_fair

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

    def generate_voucher_orders(self, voucher: Voucher, order_depth: OrderDepth,
                         position: int, signal: int) -> List[Order]:
        orders = []
        fair_value = voucher.fair_value
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        
        # Dynamic position sizing based on signal strength
        position_ratio = abs(position) / voucher.position_limit
        
        if signal == -1:  # Strong sell
            premium = max(5, int(fair_value * 0.0015))  # 0.15% or min 5
            sell_price = max(best_ask, fair_value + premium)
            qty = min(20, int(voucher.position_limit * (1 - position_ratio)))
            orders.append(Order(voucher.symbol, sell_price, -qty))
            
        elif signal == 1:  # Strong buy
            discount = max(5, int(fair_value * 0.0015))
            buy_price = min(best_bid, fair_value - discount)
            qty = min(20, int(voucher.position_limit * (1 - position_ratio)))
            orders.append(Order(voucher.symbol, buy_price, qty))
        
        # Add iceberg orders for better execution
        if abs(signal) == 1 and len(orders) > 0:
            primary_order = orders[0]
            secondary_price = primary_order.price + (-1 if signal == -1 else 1)
            secondary_qty = min(10, primary_order.quantity // 2)
            if secondary_qty > 0:
                orders.append(Order(voucher.symbol, secondary_price, 
                                -secondary_qty if signal == -1 else secondary_qty))
        
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0
        storage_costs = 0

        if "MAGNIFICENT_MACARONS" in state.observations.conversionObservations:
            macaron = self.products["MAGNIFICENT_MACARONS"]
            conv_obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            position = state.position.get("MAGNIFICENT_MACARONS", 0)

            # Update pricing model
            macaron.update_fair_value(conv_obs)

            # Calculate conversions
            conversions = macaron.calculate_conversion(conv_obs, position)

            # Apply storage costs for net long positions
            if position > 0:
                storage_costs = position * macaron.storage_cost
                # Deduct from trader's balance (implementation depends on platform)
                self.update_balance(-storage_costs)

        if "VOLCANIC_ROCK" in state.order_depths:
            rock_price = self.mid_price_fair(state.order_depths["VOLCANIC_ROCK"])
            days_left = max(1, 7 - (state.timestamp // (24 * 3600)))  # Ensure ≥1 day
            
            # Process each voucher
            for sym, voucher in [(s,p) for s,p in self.products.items() if "VOUCHER" in s]:
                if sym in state.order_depths:
                    # Update fair value and IV analysis
                    voucher.fair_value = rock_price
                    voucher.update_iv_curve(rock_price, days_left)   # base_iv
                    
                    # Get trading signal
                    signal = voucher.get_iv_signal()
                    position = state.position.get(sym, 0)
                    
                    # Generate orders based on signal
                    if signal == -1:  # Overpriced
                        orders = self.generate_voucher_orders(voucher, state.order_depths[sym],  position, "sell")
                    elif signal == 1:  # Underpriced
                        orders = self.generate_voucher_orders(voucher, state.order_depths[sym], position, "buy")
                    else:  # Neutral - market make
                        take_orders, buy_vol, sell_vol = self.take_orders(
                            voucher, state.order_depths[sym], voucher.fair_value, voucher.spread//2, position
                        )
                        make_orders, _, _ = self.make_orders(
                            voucher, state.order_depths[sym], voucher.fair_value, position, buy_vol, sell_vol
                        )
                        orders = take_orders + make_orders
                    
                    result[sym] = orders
        
        for prod_symbol, order_depth in state.order_depths.items():
            if prod_symbol in self.products and "VOUCHER" not in prod_symbol and "MAGNIFICENT_MACARON" not in prod_symbol:
                product = self.products[prod_symbol]
                position = state.position.get(prod_symbol, 0)

                product.fair_value = self.fair(prod_symbol, state, order_depth)

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

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData