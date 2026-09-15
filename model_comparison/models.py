from scipy.stats import norm, t
import numpy as np
from parameters import (
    MEAN,
    VARIANCE,
    DEGREES_OF_FREEDOM
)


# Normal to normal comparisson
def normal_model(mean, std_deviation):

    return norm(
        loc=mean,
        scale=std_deviation
    )


true_normal_dist = normal_model(MEAN, np.sqrt(VARIANCE))


# Heavy-tailed model
def heavy_tailed_model(mean, std_deviation, df=DEGREES_OF_FREEDOM):
 
    if df <= 2:
        raise ValueError("df must be > 2 for the variance to be finite")
 
    variance = std_deviation ** 2
    scale = np.sqrt(variance * (df - 2) / df)
 
    return t(
        loc=mean,
        scale=scale,
        df=df
    )


true_heavy_tailed_dist = heavy_tailed_model(MEAN, np.sqrt(VARIANCE))


class MistureModel():

    def __init__(
            self,
            pos_effect_mean,
            neg_effect_mean,
            pos_std,
            neg_std,
            null_std,
            pos_fraction,
            neg_fraction,
            seed=None
        ):
            
        self.pos_mean = pos_effect_mean
        self.neg_mean = neg_effect_mean
        self.pos_std = pos_std
        self.neg_std = neg_std
        self.null_std = null_std

        # Percentages and fractions
        self.neg_f = neg_fraction
        self.pos_f = pos_fraction
        self.null_f = 1 - self.pos_f - self.neg_f
        self.pos_p = 1 - pos_fraction
        self.neg_p = neg_fraction

        if self.neg_p > self.pos_p:
            raise ValueError(
                "Percentage of positive and negative effects was " \
                "not correct defined"
            )

        if self.neg_p < 0 or self.neg_p >= 1:
            raise ValueError(
                "Percentage of negative was not correct defined"
            )

        if self.pos_p < 0 or self.pos_p >= 1:
            raise ValueError(
                "Percentage of positive effects was not correct defined"
            )

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def rvs(self, size):

        shape = np.atleast_1d(size)
        results = np.zeros(shape, dtype=float)

        dist_choice = self.rng.uniform(0.0, 1.0, size=shape)

        neg_mask = dist_choice < self.neg_p
        null_mask = (dist_choice >= self.neg_p) & (dist_choice < self.pos_p)
        pos_mask = dist_choice >= self.pos_p

        results += neg_mask * \
            self.rng.normal(self.neg_mean, self.neg_std, shape)

        results += null_mask * \
            self.rng.normal(0, self.null_std, null_mask.sum())

        results += pos_mask * \
            self.rng.normal(self.pos_mean, self.pos_std, shape)

        return results

    def pdf(self, x):
        
        x = np.asarray(x)

        return (
            self.neg_f * norm(loc=self.neg_mean, scale=self.neg_std).pdf(x)
            + self.null_f * norm(loc=0, scale=self.null_std).pdf(x)
            + self.pos_f * norm(loc=self.pos_mean, scale=self.pos_std).pdf(x)
        )
    
    def cdf(self, x):

        x = np.asarray(x)

        return (
            self.neg_f * norm(loc=self.neg_mean, scale=self.neg_std).cdf(x)
            + self.null_f * norm(loc=0, scale=self.null_std).cdf(x)
            + self.pos_f * norm(loc=self.pos_mean, scale=self.pos_std).cdf(x)
        )