import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def prices2returns(prices):
    return np.array([(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)])

def returns2prices(p0, returns):
    if len(returns.shape) == 1:
        L = returns.shape[0]
        prices = np.zeros(L)
        prices[0] = p0 + p0 * returns[0]

        for i in range(L-1):
            prices[i+1] = prices[i] + prices[i] * returns[i]
    else:
        prices = np.array([returns2prices(p0, r) for r in returns])
    
    return prices
        
class BaseModel:
    def __init__(self, data=None):
        self.prices =           data['Close'].to_numpy()
        self.data =             data
        self.returns =          prices2returns(self.prices)
        self.pseudo_returns =   None
        self.T =                0 #T последней проведенной симуляции
        self.date_format =      r"%Y-%m-%d"
        self.dates =            [datetime.datetime.strptime(date, self.date_format).date() for date in self.data['Date'][:]]
    
    def simulate(self, steps, iterations, T=None, F=0):
        self.pseudo_returns = np.zeros([iterations, steps])
        
    def plot_last_simulation(self, N=10, padding=10, real_price=False, axis='dates'):
        steps = self.pseudo_returns.shape[1]
        if steps <= 0:  return
        p0 = self.prices[self.T-1]
        
        plt.figure(figsize=(10,5))
        plt.grid(True)
        
        end = self.T+steps if real_price else self.T 
        if axis == 'dates': 
            axis_simulated = self.dates[self.T-1:self.T+steps]
            axis_real = self.dates[self.T-padding:end]
        elif axis == 'numbers':
            axis_real = np.arange(self.T-padding, end)
            axis_simulated = np.arange(self.T-1, self.T+steps)
            
        random_indices = np.random.choice(self.pseudo_returns.shape[0], N, replace=False)
        simulated_prices = returns2prices(p0, self.pseudo_returns[random_indices, :])
        
        for sp in simulated_prices:
            sp = np.append(np.array([p0]), sp)
            plt.plot(axis_simulated, sp, color='green', linewidth=0.2)
            
        print(self.dates[self.T-1], self.dates[self.T+steps], steps)
            
        plt.plot(axis_real, self.prices[self.T-padding:end], color='blue', linewidth=2)
            
        
    def plot(self, begin=0, end=None, axis='dates'):
        if not end:     end = len(self.prices)
            
        if axis == 'dates': 
            axis = self.dates[begin:end]
        elif axis == 'numbers':
            axis = np.arange(begin,end)

        plt.figure(figsize=(10,5))
        plt.grid(True)
        plt.plot(axis, self.prices[begin:end])
    
    def VaR(self, q):
        simulated_prices = returns2prices(self.prices[self.T-1], self.pseudo_returns)
        under_q = int(q * len(simulated_prices))

        level = np.sort(simulated_prices[:,-1])[under_q + 1]
        p0 = self.prices[self.T-1]
        var = (level - p0) / p0

        return var, level
    
    def evaluate(self, start, begin, end, evaluation_step, q, steps, iterations, plot_errors=False, window=False):
        if end == None:   end = len(self.prices)
        error = 0
        total = 0
        self.T = begin
        
        for i in range(begin, end-max(steps, evaluation_step), evaluation_step):
            self.simulate(steps, iterations, F=start, T=begin)
            _, level = self.VaR(q)

            if self.prices[self.T+steps] < level: 
                error += 1
                if plot_errors: self.plot_last_simulation(50, real_price=True)

            total += 1
            begin += evaluation_step
            if window: start += evaluation_step

            
        return error/total

class RandomSampling(BaseModel):
    def __init__(self, data):
        super().__init__(data)
    
    def simulate(self, steps, iterations, F=0, T=None):
        if T == None:   T = len(self.returns)
        self.T = T

        self.pseudo_returns = np.array([
            np.random.choice(self.returns[F:self.T], steps) for _ in range(iterations)
        ])
        return self.pseudo_returns

            