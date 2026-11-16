from scipy.stats import norm, t
import numpy as np
from parameters import (
    MEAN,
    VARIANCE
)


# Normal to normal comparisson
def normal_model(mean, std_deviation):

    return norm(
        loc=mean,
        scale=std_deviation
    )


# Heavy-tailed model
def heavy_tailed_model(mean, std_deviation, df):
 
    if df <= 2:
        raise ValueError("df must be > 2 for the variance to be finite")
 
    variance = std_deviation ** 2
    scale = np.sqrt(variance * (df - 2) / df)
 
    return t(
        loc=mean,
        scale=scale,
        df=df
    )


class MistureModel():

    def __init__(
            self,
            effect_mean,
            effect_std,
            nonnull_fraction,
        ):

        if nonnull_fraction < 0 or nonnull_fraction > 1:
            raise ValueError(
                "Fraction of non-null effects was not correct defined"
            )

        # Symmetric: negative effects mirror positive ones
        self.pos_mean = effect_mean
        self.neg_mean = -effect_mean
        self.pos_std = effect_std
        self.neg_std = effect_std

        # Percentages and fractions
        self.pos_f = nonnull_fraction / 2
        self.neg_f = nonnull_fraction / 2
        self.null_f = 1 - self.pos_f - self.neg_f
        self.pos_p = 1 - self.pos_f
        self.neg_p = self.neg_f

        self.rng = np.random.default_rng()

    def rvs(self, size):

        shape = np.atleast_1d(size)
        results = np.zeros(shape, dtype=float)

        dist_choice = self.rng.uniform(0.0, 1.0, size=shape)

        neg_mask = dist_choice < self.neg_p
        pos_mask = dist_choice >= self.pos_p

        results += neg_mask * \
            self.rng.normal(self.neg_mean, self.neg_std, shape)
        results += pos_mask * \
            self.rng.normal(self.pos_mean, self.pos_std, shape)

        return results

    def pdf(self, x):

        x = np.asarray(x)

        return (
            self.neg_f * norm(loc=self.neg_mean, scale=self.neg_std).pdf(x)
            + self.pos_f * norm(loc=self.pos_mean, scale=self.pos_std).pdf(x)
        )

    def cdf(self, x):

        x = np.asarray(x)

        return (
            self.neg_f * norm(loc=self.neg_mean, scale=self.neg_std).cdf(x)
            + self.null_f * (x >= 0)
            + self.pos_f * norm(loc=self.pos_mean, scale=self.pos_std).cdf(x)
        )
    

class CenterMistureModel():

    def __init__(
            self,
            effect_std,
            null_fraction,
        ):

        if null_fraction < 0 or null_fraction > 1:
            raise ValueError(
                "Fraction of non-null effects was not correct defined"
            )

        # Symmetric: negative effects mirror positive ones
        self.eff_std = effect_std

        # Percentages and fractions
        self.null_f = null_fraction
        # Point mass at exactly 0 (has no density)
        self.atom_mass = null_fraction

        self.rng = np.random.default_rng()

    def rvs(self, size):

        shape = np.atleast_1d(size)
        results = np.zeros(shape, dtype=float)

        dist_choice = self.rng.uniform(0.0, 1.0, size=shape)

        eff_mask = dist_choice >= self.null_f

        results += eff_mask * \
            self.rng.normal(0, self.eff_std, shape)

        return results

    def pdf(self, x):

        x = np.asarray(x)

        return (
            (1 - self.null_f) * norm(loc=0, scale=self.eff_std).pdf(x)
        )

    def cdf(self, x):

        x = np.asarray(x)

        return (
            self.null_f * (x >= 0)
            + (1 - self.null_f) * norm(loc=0, scale=self.eff_std).cdf(x)
        )
    
