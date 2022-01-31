
import mat4py
import numpy as np

def loadFile(dataFile): 
    
    data = mat4py.loadmat(dataFile)
    data = [ v for v in data.values() ]
    data = data[0]
    data = np.asarray(data)
    
    return data

# Function to load the training data from matlab
def loadData(path, inputFolder, dataFile): 
    
    dataPath = path + '/' + inputFolder + '/' + dataFile
    
    data = mat4py.loadmat(dataPath)
    data = [ v for v in data.values() ]
    data = data[0]
    data = np.asarray(data)
    
    return data

def loadVocabulary(path, inputFolder, VocabularyFile):
    
    vocabularyFile = path + '/' + inputFolder + '/' + VocabularyFile
    
    vocabulary      = mat4py.loadmat(vocabularyFile)
    nClusters       = np.array(vocabulary['net']['N'])
    nodesMean       = np.array(vocabulary['net']['nodesMean'])
    dataColorNode   = np.array(vocabulary['net']['dataColorNode'])
    transitionMat   = np.array(vocabulary['net']['transitionMat'])
    transMatsTime   = np.array(vocabulary['net']['transMatsTime'])
    maxClustersTime = np.array(vocabulary['net']['maxClustersTime'])
            
    # this is a list of lists. I prefer it as a list of arrays, so we change it
    nodesCovTemp         = vocabulary['net']['nodesCov']            
    nodesCov = []
    for i in range(nClusters):
        currCov  = np.array(nodesCovTemp[i])
        nodesCov.append(currCov)
        nodesCov = nodesCov
        
    return nClusters, nodesMean, nodesCov, dataColorNode, transitionMat, transMatsTime, maxClustersTime
