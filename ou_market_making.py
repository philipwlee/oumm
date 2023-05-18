import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import stats
from abc import ABC, abstractclassmethod
import matplotlib.pylab as plt


def inverse_pdf(x):
    return np.sqrt(-2*np.log(np.sqrt(2*np.pi)*x))

def kelly_criterion(p: float | NDArray, q: float | NDArray, a: float | NDArray, b: float | NDArray) -> float | NDArray:
    return p/a - q/b

def round_to(x: float, rounder: float) -> float:
    return (x // rounder) * rounder

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
        axis.axhline(EXt1, color='b', alpha=0.3, linestyle="--")


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


class TradingStrategy(ABC):

    exposure: float

    @abstractclassmethod
    def backtest(self):
        pass

    @abstractclassmethod
    def size_position(self):
        pass


class MeanRevertingStrategy(TradingStrategy):
    """Trades a PriceProcess based on understanding the OU Process's dynamics"""

    def __init__(self,
                 pripro: PriceProcess,
                 exposure: float = 1e4,
                 z_score: float = 3,
                 fill_quality: float = 1.0):
        self.pripro = pripro
        self.exposure = exposure
        self.fill_quality = fill_quality
        self.data = pd.DataFrame()

        self.generate_sizing_factors(z_score)
        self.generate_kelly_pos()

    def backtest(self):
        self.positions = np.zeros(self.pripro.time.shape[0])

        price_arr = np.zeros(self.pripro.time.shape)
        pos_arr = np.zeros(self.pripro.time.shape)
        cash_arr = np.zeros(self.pripro.time.shape)

        last_mid = self.pripro.dynamics.X0
        last_pos = 0

        for time in self.pripro.time:
            mid = self.pripro.mid_price[time]

            state = self.data.loc[(np.isclose(self.data["MID"], mid))]
            pos = state["POS"].values[0]

            price_arr[time] = mid
            pos_arr[time] = pos

            if   last_mid < mid: ind = 1 # up move
            elif mid < last_mid: ind = -1
            else: ind = 0

            if ind==1:
                mask = ((last_mid + self.pripro.ticksize / 10 < self.data["MID"])
                         & (self.data["MID"] < mid + self.pripro.ticksize / 10))
                
                dps = self.data[mask]
                trades = dps["ASK"].values @ dps["N@ASK"].values
            elif ind==-1:
                mask = ((mid - self.pripro.ticksize / 10 < self.data["MID"])
                         & (self.data["MID"] < last_mid - self.pripro.ticksize / 10))
                
                dps = self.data[mask]
                trades = dps["BID"].values @ dps["N@BID"].values
            else:
                trades = 0

            cash_arr[time] = -trades - (1-self.fill_quality) * np.abs(last_pos - pos) * self.pripro.ticksize

            last_mid = mid
            last_pos = pos

        self.trades = pd.DataFrame({"MID": price_arr, "POS": pos_arr, "DCASH": cash_arr})
        self.trades["M2MPOS"] = self.trades["MID"] * self.trades["POS"]
        self.trades["CASH"] = self.trades["DCASH"].cumsum()
        self.trades["M2MPORT"] = self.trades["M2MPOS"] + self.trades["CASH"]

    def size_position(self, mid: float):
        pass

    def generate_sizing_factors(self, z_score: float) -> None:
        self.data["BID"] = np.arange(self.pripro.dynamics.X0 - z_score * self.pripro.dynamics.sigma,
                                     self.pripro.dynamics.X0 + z_score * self.pripro.dynamics.sigma,
                                     self.pripro.ticksize)
        self.data["MID"] = self.data["BID"] + self.pripro.ticksize / 2
        self.data["ASK"] = self.data["BID"] + self.pripro.ticksize

        self.data["EDO"] = np.zeros(self.data["MID"].values.shape)
        self.data["EUP"] = np.zeros(self.data["MID"].values.shape)
        self.data["PDO"] = np.zeros(self.data["MID"].values.shape)
        self.data["PUP"] = np.zeros(self.data["MID"].values.shape)

        for i in range(self.data["MID"].shape[0]):
            mid = self.data["MID"][i]
            bid = round_to(mid, self.pripro.ticksize)
            ask = bid + self.pripro.ticksize
            std = self.pripro.dynamics.sigma * self.pripro.dynamics.deltat**0.5

            drift = self.pripro.dynamics.theta * (self.pripro.dynamics.mu - mid) * self.pripro.dynamics.deltat

            self.data["PDO"][i] = stats.norm.cdf(bid, mid+drift, std)
            self.data["PUP"][i] = stats.norm.sf(ask,  mid+drift, std)

            self.data["EDO"][i] = mid - stats.norm.expect(loc=mid+drift, scale=std, ub=bid) / self.data["PDO"][i]
            self.data["EUP"][i] = stats.norm.expect(loc=mid+drift, scale=std, lb=ask) / self.data["PUP"][i] - mid

        self.data["TOT"] = self.data["PDO"] + self.data["PUP"]

    def generate_kelly_pos(self):
        p = self.data["PUP"] / self.data["TOT"]
        q = self.data["PDO"] / self.data["TOT"]

        b = self.data["EUP"] / np.abs(self.data["MID"])
        a = self.data["EDO"] / np.abs(self.data["MID"])

        self.data["KELLY"] = kelly_criterion(p, q, a, b)
        self.data["POS"] = np.round((self.data["KELLY"] * self.exposure))
        self.data["N@BID"] = -self.data["POS"].diff().shift(-1).fillna(0)
        self.data["N@ASK"] = self.data["POS"].diff().fillna(0)


    def plot_sizing_params(self, axis: plt.Axes):
        axis[0].plot(self.data["MID"], self.data["EDO"], color='r', label="Short given Down")
        axis[0].plot(self.data["MID"], self.data["EUP"], color='g', label="Long given Up")
        axis[0].legend()
        axis[0].set(title="Dollar Return given Outcome", xlabel="Price", ylabel="Dollar Return")

        axis[1].plot(self.data["MID"], self.data["PDO"], color='r', label="Down")
        axis[1].plot(self.data["MID"], self.data["PUP"], color='g', label="Up")
        axis[1].plot(self.data["MID"], 1-self.data["TOT"], color='b', label="Flat")
        axis[1].axhline(1)
        axis[1].legend()
        axis[1].set(title="Probability of Next Price Change", xlabel="Price", ylabel="Probability")

    def plot_sizing_array(self, axis: plt.Axes, mask: None | NDArray = None) -> None:
        if mask is None:
            mask = np.ones(self.data["KELLY"].values.shape).astype(bool)

        axis.plot(self.data["MID"][mask], self.data["KELLY"][mask])
        axis.axvline(self.pripro.dynamics.mu)
        axis.axhline(0)
        axis.set(title="Sizing According to Kelly Criterion", xlabel="Price", ylabel="Size (Multiple of Exposure)")
