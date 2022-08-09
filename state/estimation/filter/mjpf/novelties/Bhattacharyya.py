import math

import numpy as np

from mMath.probability.continous.gaussian.Gaussian import Gaussian


class Bhattacharyya:
    def getValue(self,hist1, hist2):

        # calculate mean of hist1
        h1Mean_ = np.mean(hist1)
        h1Var_ = np.var(hist1)
        h1Gs = Gaussian(h1Mean_,h1Var_)

        # calculate mean of hist2
        h2Mean_ = np.mean(hist2)
        h2Var_ = np.var(hist2)
        h2Gs = Gaussian(h2Mean_, h2Var_)
        score = np.log(math.sqrt(h1Gs,h2Gs))
        return score

    def getValue_(self,hist1, hist2):
        return 0