

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Function to create the graph structure as necessary in the code of the 
# Wasserstein graph distance
def FindGraphStructure (idx, offset):
    
    # Number of clusters
    N = int(max(idx)+1)
    
    # Length of the data
    data_length =  len(idx)
    
    # Create the index list: it just gives a correspondence key-index
    index = dict()
    for i in range (0, N):
        index[i] = i
   
    # list with all the interactions from a node i to a node j, with i != j
    interactions = []
    for i in range (0, data_length - offset):
        # If we have a movement from a cluster to another one
        #if idx[i + 1] != idx[i]:
        # Add the interaction to the list
        pair = [idx[i + offset], idx[i]]
        interactions.append(pair)
    # Sort the list in ascending order of numbers
    interactions.sort() 
    
    return index, interactions

def FindTransitionMat (index, interactions, result_folder, plotting):
    
    interactionMat = np.zeros((len(index), len(index)))
    for item in interactions:
        interactionMat[item[0], item[1]] += 1
        
    if plotting == True:
        plt.imshow(interactionMat)
        plt.savefig(result_folder)
        plt.close('all')
            
    interactionMat = normalize(interactionMat, axis=1, norm='l1')
        
    return interactionMat

def BuildVocabulary(nClusters, lengthOfData, dataColorNode, data, dataNorm, numberLatentStates, z_dim):
    
    # Define one array for each cluster:
    datanodes = []
    for i in range(nClusters):
        cluster_i = []
        datanodes.append(cluster_i)
        
    # Insert the data of each cluster in the corresponding array
    for i in range(lengthOfData):
        superstate_i = int(dataColorNode[i])
        state_i = data[i, :]
        datanodes[superstate_i].append(state_i)
    
    datanodesNorm = []
    for i in range(nClusters):
        cluster_i_norm = []
        datanodesNorm.append(cluster_i_norm)
        
    for i in range(lengthOfData):
        superstate_i_norm = int(dataColorNode[i])
        state_i_norm = dataNorm[i, :]
        datanodesNorm[superstate_i_norm].append(state_i_norm)
    
    summing = 0
    for i in range(len(datanodes)):
        summing += len(datanodes[i])
    
    # Calculate the covariance for each array
    covSuperstates = np.zeros((nClusters, numberLatentStates*2, numberLatentStates*2))            
            
    #--------------------------------------------------------------------------
    # Radius of acceptance and covariance
    
    nodesCov = []
    nodesCovNorm = []
    for i in range(nClusters):
        nodesCov_i = []
        nodesCov.append(nodesCov_i)
        #nodesCovNorm.append(nodesCov_i)
    nodesRadAccepts_state = []    
    nodesRadAcceptNorms_state = []
    nodesRadAccepts_deriv = []  
    nodesRadAcceptNorms_deriv = []
    
        
    for i in range(nClusters):
        
        if len(datanodes[i]) == 0:
            nodesCov[i].append(np.zeros((numberLatentStates, numberLatentStates)))
            
        else:        
            datanode_of_cluster = np.asarray(datanodes[i])
            datanode_of_cluster_norm = np.asarray(datanodesNorm[i])
            
            nodesCov[i] = np.cov(np.transpose(datanode_of_cluster)) # calculation of covariance values
            #nodesCovNorm[i] = np.cov(np.transpose(datanode_of_cluster))  
            
        # Calculation of radius of acceptance
        nodeRadAccept_state = np.sqrt(np.sum(np.power((3*np.std(datanode_of_cluster[:,0:z_dim], axis = 0)),2)))
        nodeRadAcceptNorm_state = np.sqrt(np.mean(np.power((3*np.std(datanode_of_cluster[:,0:z_dim], axis = 0)),2)))
        
        
        nodeRadAccept_state = np.sqrt(np.sum(np.power((3*np.std(datanode_of_cluster[:, 0:z_dim], axis = 0)),2)))
        nodeRadAcceptNorm_state = np.sqrt(np.mean(np.power((3*np.std(datanode_of_cluster_norm[:, 0:z_dim], axis = 0)),2)))
        
        nodeRadAccept_deriv = np.sqrt(np.mean(np.power((3*np.std(datanode_of_cluster[:,z_dim:z_dim*2] , axis = 0)),2)))
        nodeRadAcceptNorm_deriv = np.sqrt(np.mean(np.power((3*np.std(datanode_of_cluster_norm[:,z_dim:z_dim*2], axis = 0)),2)))
        
        nodesRadAccepts_state.append(nodeRadAccept_state)
        nodesRadAcceptNorms_state.append(nodeRadAcceptNorm_state)
        
        nodesRadAccepts_deriv.append(nodeRadAccept_deriv)
        nodesRadAcceptNorms_deriv.append(nodeRadAcceptNorm_deriv)
        
    
    covSuperstates = nodesCov
    radiusState = nodesRadAccepts_state
    radiusStateNorm = nodesRadAcceptNorms_state
    
    radiusDeriv = nodesRadAccepts_deriv
    radiusDerivNorm = nodesRadAcceptNorms_deriv
    

    
    index, interactions = FindGraphStructure (dataColorNode, 1)
    transitionMat = FindTransitionMat(index, interactions, " ", plotting = False)
    
    return covSuperstates, nodesCov, covSuperstates, radiusState, radiusStateNorm, radiusDeriv, radiusDerivNorm, transitionMat