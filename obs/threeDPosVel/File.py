import numpy as np


class File:
    '''

    '''
    def __init__(self, filePath: str):
        ''''''
        self.__filePath = filePath

    def getNpArr(self, maxRowsNum=None)->np.ndarray:
        ''''''
        file = open(self.__filePath, "r")
        arr = np.empty((0, 6), np.dtype('float64'))
        for lineCounter, curLine in enumerate(file):
            if maxRowsNum is not None and lineCounter > maxRowsNum:
                break
            arr = np.append(arr, np.asarray([[r for r in curLine.split(",")]]), axis=0)
        opt = arr.astype(np.float)
        file.close()
        return np.asarray(opt)
