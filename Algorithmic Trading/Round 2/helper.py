# def get_synthetic_basket2_order_depth(
#         self, order_depths: Dict[str, OrderDepth]
#     ) -> OrderDepth:
#         # Constants
#         CROISSANTS_PER_BASKET = PICNIC_BASKET2_WEIGHTS[Product.CROISSANTS]
#         JAMS_PER_BASKET = PICNIC_BASKET2_WEIGHTS[Product.JAMS]
#         DJEMBES_PER_BASKET = PICNIC_BASKET2_WEIGHTS[Product.DJEMBES]

#         # Initialize the synthetic basket order depth
#         synthetic_order_price = OrderDepth()

#         # Calculate the best bid and ask for each component
#         croissants_best_bid = (
#             max(order_depths[Product.CROISSANTS].buy_orders.keys())
#             if order_depths[Product.CROISSANTS].buy_orders
#             else 0
#         )
#         croissants_best_ask = (
#             min(order_depths[Product.CROISSANTS].sell_orders.keys())
#             if order_depths[Product.CROISSANTS].sell_orders
#             else float("inf")
#         )
#         jams_best_bid = (
#             max(order_depths[Product.JAMS].buy_orders.keys())
#             if order_depths[Product.JAMS].buy_orders
#             else 0
#         )
#         jams_best_ask = (
#             min(order_depths[Product.JAMS].sell_orders.keys())
#             if order_depths[Product.JAMS].sell_orders
#             else float("inf")
#         )
#         djembes_best_bid = (
#             max(order_depths[Product.DJEMBES].buy_orders.keys())
#             if order_depths[Product.DJEMBES].buy_orders
#             else 0
#         )
#         djembes_best_ask = (
#             min(order_depths[Product.DJEMBES].sell_orders.keys())
#             if order_depths[Product.DJEMBES].sell_orders
#             else float("inf")
#         )

#         # Calculate the implied bid and ask for the synthetic basket
#         implied_bid = (
#             croissants_best_bid * CROISSANTS_PER_BASKET
#             + jams_best_bid * JAMS_PER_BASKET
#             + djembes_best_bid * DJEMBES_PER_BASKET
#         )
#         implied_ask = (
#             croissants_best_ask * CROISSANTS_PER_BASKET
#             + jams_best_ask * JAMS_PER_BASKET
#             + djembes_best_ask * DJEMBES_PER_BASKET
#         )

#         # Calculate the maximum number of synthetic baskets available at the implied bid and ask
#         if implied_bid > 0:
#             croissants_bid_volume = (
#                 order_depths[Product.CROISSANTS].buy_orders[croissants_best_bid]
#                 // CROISSANTS_PER_BASKET
#             )
#             jams_bid_volume = (
#                 order_depths[Product.JAMS].buy_orders[jams_best_bid]
#                 // JAMS_PER_BASKET
#             )
#             djembes_bid_volume = (
#                 order_depths[Product.DJEMBES].buy_orders[djembes_best_bid]
#                 // DJEMBES_PER_BASKET
#             )
#             implied_bid_volume = min(
#                 croissants_bid_volume, jams_bid_volume, djembes_bid_volume
#             )
#             synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

#         if implied_ask < float("inf"):
#             croissants_ask_volume = (
#                 -order_depths[Product.CROISSANTS].sell_orders[croissants_best_ask]
#                 // CROISSANTS_PER_BASKET
#             )
#             jams_ask_volume = (
#                 -order_depths[Product.JAMS].sell_orders[jams_best_ask]
#                 // JAMS_PER_BASKET
#             )
#             djembes_ask_volume = (
#                 -order_depths[Product.DJEMBES].sell_orders[djembes_best_ask]
#                 // DJEMBES_PER_BASKET
#             )
#             implied_ask_volume = min(
#                 croissants_ask_volume, jams_ask_volume, djembes_ask_volume
#             )
#             synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

#         return synthetic_order_price
    
#     def convert_synthetic_basket2_orders(
#         self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
#     ) -> Dict[str, List[Order]]:
#         # Initialize the dictionary to store component orders
#         component_orders = {
#             Product.CROISSANTS: [],
#             Product.JAMS: [],
#             Product.DJEMBES: [],
#         }

#         # Get the best bid and ask for the synthetic basket
#         synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(
#             order_depths
#         )
#         best_bid = (
#             max(synthetic_basket_order_depth.buy_orders.keys())
#             if synthetic_basket_order_depth.buy_orders
#             else 0
#         )
#         best_ask = (
#             min(synthetic_basket_order_depth.sell_orders.keys())
#             if synthetic_basket_order_depth.sell_orders
#             else float("inf")
#         )

#         # Iterate through each synthetic basket order
#         for order in synthetic_orders:
#             # Extract the price and quantity from the synthetic basket order
#             price = order.price
#             quantity = order.quantity

#             # Check if the synthetic basket order aligns with the best bid or ask
#             if quantity > 0 and price >= best_ask:
#                 # Buy order - trade components at their best ask prices
#                 croissants_price = min(
#                     order_depths[Product.CROISSANTS].sell_orders.keys()
#                 )
#                 jams_price = min(
#                     order_depths[Product.JAMS].sell_orders.keys()
#                 )
#                 djembes_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
#             elif quantity < 0 and price <= best_bid:
#                 # Sell order - trade components at their best bid prices
#                 croissants_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
#                 jams_price = max(
#                     order_depths[Product.JAMS].buy_orders.keys()
#                 )
#                 djembes_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
#             else:
#                 # The synthetic basket order does not align with the best bid or ask
#                 continue

#             # Create orders for each component
#             croissants_order = Order(
#                 Product.CROISSANTS,
#                 croissants_price,
#                 quantity * PICNIC_BASKET2_WEIGHTS[Product.CROISSANTS],
#             )
#             jams_order = Order(
#                 Product.JAMS,
#                 jams_price,
#                 quantity * PICNIC_BASKET2_WEIGHTS[Product.JAMS],
#             )
#             djembes_order = Order(
#                 Product.DJEMBES, djembes_price, quantity * PICNIC_BASKET2_WEIGHTS[Product.DJEMBES]
#             )

#             # Add the component orders to the respective lists
#             component_orders[Product.CROISSANTS].append(croissants_order)
#             component_orders[Product.JAMS].append(jams_order)
#             component_orders[Product.DJEMBES].append(djembes_order)

#         return component_orders
    
#     def execute_spread2_orders(
#         self,
#         target_position: int,
#         basket_position2: int,
#         order_depths: Dict[str, OrderDepth],
#     ):

#         if target_position == basket_position2:
#             return None

#         target_quantity = abs(target_position - basket_position2)
#         basket_order_depth = order_depths[Product.PICNIC_BASKET2]
#         synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

#         if target_position > basket_position2:
#             basket_ask_price = min(basket_order_depth.sell_orders.keys())
#             basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

#             synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
#             synthetic_bid_volume = abs(
#                 synthetic_order_depth.buy_orders[synthetic_bid_price]
#             )

#             orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
#             execute_volume = min(orderbook_volume, target_quantity)

#             basket_orders = [
#                 Order(Product.PICNIC_BASKET2, basket_ask_price, execute_volume)
#             ]
#             synthetic_orders = [
#                 Order(Product.SYNTHETIC2, synthetic_bid_price, -execute_volume)
#             ]

#             aggregate_orders = self.convert_synthetic_basket2_orders(
#                 synthetic_orders, order_depths
#             )
#             aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
#             return aggregate_orders

#         else:
#             basket_bid_price = max(basket_order_depth.buy_orders.keys())
#             basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

#             synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
#             synthetic_ask_volume = abs(
#                 synthetic_order_depth.sell_orders[synthetic_ask_price]
#             )

#             orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
#             execute_volume = min(orderbook_volume, target_quantity)

#             basket_orders = [
#                 Order(Product.PICNIC_BASKET2, basket_bid_price, -execute_volume)
#             ]
#             synthetic_orders = [
#                 Order(Product.SYNTHETIC2, synthetic_ask_price, execute_volume)
#             ]

#             aggregate_orders = self.convert_synthetic_basket2_orders(
#                 synthetic_orders, order_depths
#             )
#             aggregate_orders[Product.PICNIC_BASKET2] = basket_orders
#             return aggregate_orders

#     def spread2_orders(
#         self,
#         order_depths: Dict[str, OrderDepth],
#         product: Product,
#         basket_position2: int,
#         spread_data: Dict[str, Any],
#     ):
#         if Product.PICNIC_BASKET2 not in order_depths.keys():
#             return None

#         basket_order_depth = order_depths[Product.PICNIC_BASKET2]
#         synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
#         basket_swmid = self.get_swmid(basket_order_depth)
#         synthetic_swmid = self.get_swmid(synthetic_order_depth)
#         spread = basket_swmid - synthetic_swmid
#         spread_data["spread_history"].append(spread)

#         if (
#             len(spread_data["spread_history"])
#             < self.params[Product.SPREAD2]["spread_std_window"]
#         ):
#             return None
#         elif len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
#             spread_data["spread_history"].pop(0)

#         spread_std = np.std(spread_data["spread_history"])

#         zscore = (
#             spread - self.params[Product.SPREAD2]["default_spread_mean"]
#         ) / spread_std

#         if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
#             if basket_position2 != -self.params[Product.SPREAD2]["target_position"]:
#                 return self.execute_spread2_orders(
#                     -self.params[Product.SPREAD2]["target_position"],
#                     basket_position2,
#                     order_depths,
#                 )

#         if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
#             if basket_position2 != self.params[Product.SPREAD2]["target_position"]:
#                 return self.execute_spread2_orders(
#                     self.params[Product.SPREAD2]["target_position"],
#                     basket_position2,
#                     order_depths,
#                 )

#         spread_data["prev_zscore"] = zscore
#         return None

