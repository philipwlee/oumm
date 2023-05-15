import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractclassmethod
import matplotlib.pylab as plt


def inverse_pdf(x):
    return np.sqrt(-2*np.log(np.sqrt(2*np.pi)*x))

def kelly_criterion(p: float, q: float, a: float, b: float) -> float:
    return p/a - q/b

class Dynamics(ABC):

    name: str

    @abstractclassmethod
    def generate(self):
        pass

    @abstractclassmethod
    def plot_next_pdf(self):
        pass

class OrnsteinUhlenbeck(Dynamics):
    """Object to simulate a mean reverting Brownian motion"""

    X0: float
    theta: float
    mu: float
    sigma: float
    deltat: float
    name: str = "Ornstein-Uhlenbeck Process"

    def __init__(self, X0: float, theta: float, mu: float, sigma: float, deltat: float) -> None:
        """Sets the necessary variables for the OU process"""
        self.X0 = X0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.deltat = deltat

    def generate(self, n_steps: int, n_paths: int = 1) -> NDArray:
        """Creates a time series of OU paths by iteratively updating"""

        # Set up grid of Xt values
        dBt = np.random.normal(0, 1, (n_steps, n_paths))
        Xt = self.X0 * np.ones((n_steps, n_paths))

        # i is the time index
        for i in range(n_steps-1):
            DeltaXti = self.theta * (self.mu - Xt[i]) * self.deltat + self.sigma * self.deltat**0.5 * dBt[i]
            Xt[i+1] = Xt[i] + DeltaXti

        return Xt
    
    def calc_drift(self, Xt: NDArray, theta: float, mu: float, deltat: float) -> NDArray:
        return theta * (mu - Xt) * deltat
    
    def plot_next_pdf(self, axis: plt.Axes, Xt: float) -> None:
        xs = np.linspace(1/np.sqrt(2*np.pi), 1e-6, 100)
        ys = inverse_pdf(xs)

        EXt1 = Xt + self.calc_drift(Xt, self.theta, self.mu, self.deltat)
        axis.plot(xs, ys*self.sigma*self.deltat**0.5 + EXt1, color='b')
        axis.plot(xs, EXt1-ys*self.sigma*self.deltat**0.5, color='b')
        axis.invert_xaxis()
        axis.axhline(EXt1, color='b', alpha=0.3)


class PriceProcess:
    """Uses dynamics to generate latent price and create discretized bid/ask"""

    def __init__(self, dynamics: Dynamics, ticksize: float) -> None:
        self.dynamics = dynamics
        self.ticksize = ticksize

    def generate(self, n_steps: int) -> None:
        self.time = np.arange(0, n_steps, 1)
        # Single path of price process
        self.latent_price = self.dynamics.generate(n_steps).reshape(self.time.shape)
        self.bid_price = (self.latent_price // self.ticksize) * self.ticksize
        self.ask_price = self.bid_price + self.ticksize
        self.mid_price = self.bid_price + self.ticksize / 2

    def _validate_slice(self, slice: NDArray | None) -> NDArray:
        if slice is None:
            slice = np.ones(self.time.shape[0]).astype(bool)
        return slice

    def plot_latent_price(self, axis: plt.Axes, slice: NDArray | None = None) -> None:
        slice = self._validate_slice(slice)

        axis.step(self.time[slice], self.latent_price[slice], label="Latent Price")
        axis.set(xlabel="Time", ylabel="Price", title=f"Latent Price ({self.dynamics.name})")

    def plot_bid_ask(self, axis: plt.Axes, slice: NDArray | None = None) -> None:
        slice = self._validate_slice(slice)

        axis.step(self.time[slice], self.bid_price[slice], color='g', label="Bid")
        axis.step(self.time[slice], self.latent_price[slice], color='b', alpha=0.3, label="Latent Price")
        axis.step(self.time[slice], self.ask_price[slice], color='r', label="Ask")

        axis.set(xlabel="Time", ylabel="Price", title=f"Discretized vs Latent Price ({self.dynamics.name})")
        axis.legend()


