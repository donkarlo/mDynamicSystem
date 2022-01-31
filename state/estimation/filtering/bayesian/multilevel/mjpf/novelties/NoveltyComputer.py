import numpy as np
class NoveltyComputer:

    def getNoveltyValue(self,coefficient=1)->float:
        # for i in range (1,1000000):
        #     value =  np.random.rand(1, 1)[0]
        return coefficient*np.random.rand(1, 1)[0][0]