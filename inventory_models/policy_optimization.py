import numpy as np
from scipy.stats import norm


class InventoryOptimizer:
    """Class for inventory optimization."""

    def __init__(self, cost_params):
        self.cost_params = cost_params

    def calculate_cost(self, demand, inventory_level, h, s):
        """
        lost sales cost function, for one item one period
        cost includes: holding + shortage
        """
        h_cost = h * max(0, (inventory_level - demand))
        s_cost = s * max(0, (demand - inventory_level))
        cost = h_cost + s_cost
        # print(f'shortage units: {max(0, round(demand - inventory_level))}')
        # print(f'holding units: {max(0, round(inventory_level - demand))}')
        return cost, h_cost, s_cost

    def compute_policy(self, forecast, sigma):
        """
        Compute Optimal policy (V)ARMA demand process
        """
        n = forecast.shape[1]
        T = forecast.shape[0]
        optimal_policy = np.zeros((T, n))
        y_opt = np.zeros((n))
        h = self.cost_params['holding_cost']
        s = self.cost_params['shortage_cost']
        for t in range(T):
            for i in range(n):
                p = s[i]/(s[i]+h[i])
                Z_0 = norm.ppf(p)

                y_opt[i] = forecast[t, i] + Z_0 * sigma[i]
            optimal_policy[t, :] = y_opt

        return optimal_policy

    def evaluate_policy(self, demand, optimal_policy):
        """
        evaluate policy, the cost for all periods and items, given the actual demand nd optimal policy

        output: total cost
        """
        n = demand.shape[1]
        T = demand.shape[0]

        inventory_level = np.zeros((T, n))
        order_decision = np.zeros((T, n))
        h = self.cost_params['holding_cost']
        s = self.cost_params['shortage_cost']
        total_cost = 0#np.zeros((n))
        h_cost_t = 0
        s_cost_t = 0
        
        for t in range(T):

            # Place order using a policy (base stock policy)
            for i in range(n):
                if inventory_level[t, i] <= optimal_policy[t, i]:
                    order_decision[t, i] = 1
                    inventory_level[t, i] = optimal_policy[t, i]
                # Update inventory levels at the end of period t (start of period t+1) lag delivery=0
                if t < T-1:
                    inventory_level[t+1, i] = max(inventory_level[t, i] - demand[t, i], 0)
                else:
                    continue

                # Update the cost at the end of period for each item
                cost, h_cost, s_cost = self.calculate_cost(demand[t, i],
                                                     inventory_level[t, i], h[i], s[i])
                h_cost_t += h_cost
                s_cost_t += s_cost
                total_cost += cost
        
        cost_ratio = h_cost_t and s_cost_t/h_cost_t or 0
        print(f's={s} and h={h} -> shortage/holding total cost ratio = {cost_ratio:.2f}',flush=True) 
        print(f'             -> shortage= {s_cost_t:.2f}, holding={h_cost_t:.2f}',flush=True) 
        return total_cost, h_cost_t,s_cost_t


# test usage

# n = 2
# T = 10
# h = [1, 1]
# s = [1, 1]
# cost_params = {
#     "holding_cost": h,
#     "shortage_cost": s
# }
# inv_opt = InventoryOptimizer(cost_params)
# sigma = [1, 1]
# forecast = np.ones((T, n))
# demand = np.ones((T, n))
# optimal_policy = inv_opt.compute_policy(forecast, sigma)
# evaluate_policy = inv_opt.evaluate_policy(demand, optimal_policy)
