
# This script contains functions for the ClusteringGraph CLASS
# This CLASS can be used to train a GNG, obtain a clustering graph, or read
# a clustering graph.

import numpy as np
import mat4py

from scipy.io import savemat

import torch
import copy 

import Distance_utils as d_utils
import Config_GPU     as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ---------------------------  ClusteringGraph   ------------------------------
###############################################################################

class ClusteringGraph(object):

    def Bring_nodesCov_from_list_to_array(self, nodesCovList):
        
        nodesCov = []
        for i in range(self.num_clusters):
            currCov  = np.array(nodesCovList[i])
            nodesCov.append(currCov)
            
        return nodesCov

    # Extracting the features related to training clustering
    # input:  KVAE (self)
    # output: KVAE (self) (with features extracted)
    # Extracted features:
    # - num_clusters: number of clusters
    # - nodesMean: mean value of each node (e.g., containing mean position/velocity/etc. of every node)
    # - data_m: synchronized data
    # - nodesCov: covariance of each node
    def Extract_clustering_data_features(self, clustering_data, nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        # Extracting the number of clusters, mean of clusters, and synchronized data
        self.num_clusters     = np.array(clustering_data['net']['N'])
        self.nodesMean        = np.array(clustering_data['net'][nodesMeanName])
        #self.data_m          = np.array(clustering_data['net']['data_sync'])
        clustersSequence      = np.array(clustering_data['net']['dataColorNode'])
        
        # shift cluster sequence so that first cluster is always 0
        # (as MATLAB starts from 1 instead)
        firstClusterIndex     = np.min(clustersSequence)
        clustersSequence      = clustersSequence - firstClusterIndex
        self.clustersSequence = np.squeeze(clustersSequence)
    
        # Extracting the cluster covariances ...
        nodesCovTemp    = clustering_data['net'][nodesCovName] 
        # ... and putting them in a numpy array instead of a list           
        self.nodesCov   = self.Bring_nodesCov_from_list_to_array(nodesCovTemp)
        
        # If there are additional info in the cluster
        if 'additionalClusterInfo' in clustering_data['net'].keys():
            self.additionalClusterInfo = np.array(clustering_data['net']['additionalClusterInfo'])
        
        self.CalculateAllTransitionMatrices()
        
        #######################################################################
        # Additional features to the base one      
        if 'nodesCovPred' in clustering_data['net'].keys():
            nodesCovPredTemp  = clustering_data['net']['nodesCovPred'] 
            self.nodesCovPred = self.Bring_nodesCov_from_list_to_array(nodesCovPredTemp)
        else:
            self.nodesCovPred = None
        if 'nodesCovD' in clustering_data['net'].keys():
            nodesCovDTemp     = clustering_data['net']['nodesCovD'] 
            self.nodesCovD    = self.Bring_nodesCov_from_list_to_array(nodesCovDTemp)
        else:
            self.nodesCovD    = None
        
        return
    
    def Extract_mean_std_of_data(self, clusteringFile):
        
        clustering_data = mat4py.loadmat(clusteringFile)
        
        self.X_mean = np.array(clustering_data['net']['X_mean'])
        self.X_std  = np.array(clustering_data['net']['X_std'])
        
        return
    
    def Assign_clustering_data_features(self, num_clusters, nodesMean, clustersSequence, nodesCov):
        
        self.num_clusters     = num_clusters
        self.nodesMean        = nodesMean
        self.clustersSequence = clustersSequence
        self.nodesCov         = nodesCov
        
        self.CalculateAllTransitionMatrices()

        return
    
    # This is a transition matrix over the second half of the temporal 
    # transition matrices space. This is inserted in order to avoid overfitting
    # w.r.t. the training case when temporal transition matrices are used.
    # If, for example, we have 130 temporal transition matrices rows related to
    # the passage from cluster 6 to the other ones and in that case there was
    # a passage from cluster 6 to cluster 9, if we use that transition matrix
    # for every time that we have been 130 time instants in cluster 6 or more
    # than 130 time instants in cluster 6, we will do that prediction for all
    # particles. If in the testing phase that passage does not happen,
    # we will end up having very high anomaly. In the odometry from video case
    # this will be much worse because we will loose track of where we were!!
    def CalculateSecondHalfTransitionMatrix(self):
        
        return
    
    # Temporal transition matrices averaged over a window to avoid overfitting
    def CalculateWindowedTemporalTransitionMatrices(self, window_percentage = 0.1):
        
        # Total length of training data
        numberTrainingData    = self.clustersSequence.shape[0]
        # Find max number of time instants before a zone change 
        tMax                  = self.FindMaxTimeBeforeTransition()
        # Maximum time spent in each cluster
        maxClustersTime       = self.maxClustersTime
        
        # Initialize
        windowedtransMatsTime = []
        for i in range(tMax):
            windowedtransMatsTime.append(np.zeros((self.num_clusters,
                                                   self.num_clusters)))
        
        # I take the first zone in the trajectory
        prevZone              = self.clustersSequence[0]
        time = -1
        
        # loop over the number of time instants of the trajectory
        for t in range(1, numberTrainingData):
            
            # I add a time instant
            time = time + 1
            
            # New zone
            newZone                  = self.clustersSequence[t]
            
            # I increment the corresponding value in the correct transition matrices
            # I do this over a window, so first I have to find the window
            maxTimeForCurrentCluster = maxClustersTime[prevZone]
            beginWindow              = time - maxTimeForCurrentCluster*window_percentage
            beginWindow              = int(np.floor(max(0, beginWindow))) # avoid going negative
            endWindow                = time + maxTimeForCurrentCluster*window_percentage
            endWindow                = int(np.floor(min(maxTimeForCurrentCluster, endWindow)))
            
            for w in range(beginWindow, endWindow):
                windowedtransMatsTime[w][prevZone, newZone] += 1
            
            # If I change the zone with respect to the previous time instant(/s)
            if (prevZone != newZone):
                # And I update the zone value
                prevZone = newZone
                # I reinitialize the time
                time = -1
        
        # For each transition matrix
        for t in range(tMax):
            # looping over rows of current matrix
            for row in range(self.num_clusters):
                # sum of the elements of the row
                sumElementsRow = np.sum(windowedtransMatsTime[t][row, :])
                
                # to prevent division by 0
                if sumElementsRow != 0:
                    # looping over columns of current matrix
                    for column in range(self.num_clusters):
                        # normalise matrix element
                        windowedtransMatsTime[t][row, column] = \
                            windowedtransMatsTime[t][row, column]/sumElementsRow
                            
        self.windowedtransMatsTime = windowedtransMatsTime
            
        return
    
    def CalculateAllTransitionMatrices(self):
        
        # Find transition matrix
        self.FindTransitionMatrix()
        
        # Find temporal transition matrices
        self.CalculateTemporalTransitionMatrices()
        
        self.CalculateMaxClustersTime()
        
        self.CalculateWindowedTemporalTransitionMatrices()
        
        return
    
    def FindMaxTimeBeforeTransition(self):

        # Initialise the max to 1
        tMax = 1
        # Initialise the variable containing the previous zone of a comparison
        # with the first zone of the trajectory
        oldZone = self.clustersSequence[0]  
        # Initialise the length of the run
        currentMax = 1
            
        # Looping over the time instants of the trajectory
        timeInstantsCurrentTrajectory = self.clustersSequence.shape[0]
        for t in range( 1, timeInstantsCurrentTrajectory):
        
            # Zone at current time
            newZone = self.clustersSequence[t]
        
            # If the zone has not changed
            if newZone == oldZone:
                # Increment the max of current run
                currentMax = currentMax + 1
            # if the zone has changed
            else:
                oldZone = newZone;
                # Check if a longer run has been foung
                if currentMax > tMax:
                    tMax = currentMax   
                # Reinitialize current max
                currentMax = 1
                
        return tMax
    
    def FindTimeTransitionMatrices(self, tMax, numberTrainingData):


        transMatsTime = []
        for i in range(tMax):
            transMatsTime.append(np.zeros((self.num_clusters,self.num_clusters)))
        
        # I take the first zone in the trajectory
        prevZone = self.clustersSequence[0]
        time = -1
        
        # loop over the number of time instants of the trajectory
        for t in range(1, numberTrainingData):
            # I add a time instant
            time = time + 1
            if (time > tMax):
                break
            
            # New zone
            newZone = self.clustersSequence[t]
            # If I change the zone with respect to the previous time instant(/s)
            if (prevZone != newZone):
                # I increment the corresponding value in the correct transition matrix
                transMatsTime[time][prevZone, newZone] += 1
                # And I update the zone value
                prevZone = newZone
                # I reinitialize the time
                time = -1
            # Otherwise, if I remain in the same zone
            else:
                transMatsTime[time][prevZone, newZone] += 1;
        
        # For each transition matrix
        for t in range(tMax):
            # looping over rows of current matrix
            for row in range(self.num_clusters):
                # sum of the elements of the row
                sumElementsRow = np.sum(transMatsTime[t][row, :])
                
                # to prevent division by 0
                if sumElementsRow != 0:
                    # looping over columns of current matrix
                    for column in range(self.num_clusters):
                        # normalise matrix element
                        transMatsTime[t][row, column] = \
                            transMatsTime[t][row, column]/sumElementsRow
                            
        self.transMatsTime = transMatsTime
        
        return
    
    def CalculateMaxClustersTime(self):

        # Total length of training data
        currLength    = self.clustersSequence.shape[0]
        
        # Where to insert the max time values
        maxClustersTime = np.zeros((self.num_clusters,))
        
        # Initialise the variable containing the previous zone of a comparison
        # with the first zone of the trajectory
        oldZone   = self.clustersSequence[0]
        finalZone = self.clustersSequence[currLength-1]
            
        # Initialise the length of the run
        currentMax  = 0
        currentMax2 = 0
        for t in range(currLength):
            
            # Zone at current time
            newZone = self.clustersSequence[t]
            
            # If the zone has not changed
            if newZone != finalZone:
                if newZone == oldZone:
                    # Increment the max of current run
                    currentMax = currentMax + 1
                # if the zone has changed
                else:
                    # Check if a longer run has been foung
                    if currentMax > maxClustersTime[oldZone]:
                        maxClustersTime[oldZone] = int(currentMax)
                    oldZone = newZone;
                    # Reinitialize current max
                    currentMax = 1
            else:
                if currentMax > maxClustersTime[oldZone]:
                    maxClustersTime[oldZone] = int(currentMax)
                # Increment the max of current run
                currentMax2 = currentMax2 + 1
                if currentMax > maxClustersTime[newZone]:
                    maxClustersTime[newZone] = int(currentMax2)

        self.maxClustersTime = maxClustersTime
        
        return

    
    # Input and Output: net structure obtained with GNG clustering
    # The function creates a set of transition matrices, one for each time value
    # from t to tMax, being tMax the time we have already spent in a node. 
    def CalculateTemporalTransitionMatrices(self):
    
        ## Temporal Transition Matrix
        
        # Total length of training data
        currLength    = self.clustersSequence.shape[0]
        
        # Find max number of time instants before a zone change 
        tMax          = self.FindMaxTimeBeforeTransition()
        
        # Find TIME TRANSITION MATRICES for follower
        self.FindTimeTransitionMatrices(tMax, currLength)
        
        return
    
    # Initialize the clustering with two nodes
    # input: - z_dim: dimension of latent states
    #        - low_value_z, low_value_z_dev (optional): min values of 
    #          z and of its derivative
    #        - high_value_z, high_value_z_dev (optional): max values of 
    #          z and of its derivative
    def __init__(self, 
                 z_dim, 
                 low_value_z  = -1, low_value_z_dev  = -0.01, 
                 high_value_z =  1, high_value_z_dev =  0.01):
        
        # Initial 2 nodes for training the algorithm
        self.num_clusters = 2                                                                    
        
        # Mean of nodes
        self.nodesMean = np.zeros((self.num_clusters, z_dim*2))
        
        # It returns an array of random numbers generated from the continuous uniform 
        for i in range(self.num_clusters):
            for j in range(z_dim*2):
                
                if j < z_dim:
                    self.nodesMean[i,j] = np.random.uniform(low = low_value_z    , high = high_value_z    , size=None)   
                    
                else:
                    self.nodesMean[i,j] = np.random.uniform(low = low_value_z_dev, high = high_value_z_dev, size=None) 
                    
        # Don't put any clusters sequence yet
        self.clustersSequence = 0                           

        return
    
    # Function to inizialize to zero the graph connections (to use at beginning
    # of GNG training).
    def InitializeGraphConnections(self):
        
        # error
        self.E       = np.zeros(self.num_clusters)
        # mUtility
        self.utility = np.ones(self.num_clusters)
        # Connections between nodes
        self.C       = np.zeros((self.num_clusters, self.num_clusters))
        # Ages of the edges
        self.t       = np.zeros((self.num_clusters, self.num_clusters))
        
        return
    
    # Initialize the clustering by reading it from a MATLAB file
    def LoadGraphFromMATLAB(self, clusteringFile, nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        clustering_data = mat4py.loadmat(clusteringFile)
            
        # Extract the features related to clustering
        self.Extract_clustering_data_features(clustering_data, nodesMeanName, nodesCovName)
        
        return
    
    def FindDataForEachCluster(self, data):
        
        lengthOfData = data.shape[0]
        
        # Define one array for each cluster:
        datanodes = []
        for i in range(self.num_clusters):
            cluster_i = []
            datanodes.append(cluster_i)
            
        # Insert the data of each cluster in the corresponding array
        for i in range(lengthOfData):
            superstate_i = int(self.clustersSequence[i])
            state_i = data[i, :]
            datanodes[superstate_i].append(state_i)
            
        return datanodes
    
    def FindCovarianceForEachCluster(self, datanodes, dimensionState):
        
        # Find the covariance for each cluster
        nodesCov = []
        for i in range(self.num_clusters):
            nodesCov_i = []
            nodesCov.append(nodesCov_i)
            
        # Looping over the number of clusters
        for i in range(self.num_clusters):
            # If there is no data in that node
            if len(datanodes[i]) == 0:
                nodesCov[i].append(np.zeros((dimensionState, dimensionState)))
            # If there is data in the node
            else:        
                datanode_of_cluster = np.asarray(datanodes[i])
                nodesCov[i] = np.cov(np.transpose(datanode_of_cluster))
                
        self.nodesCov = nodesCov
        
        return
    
    def FindMeanForEachCluster(self, datanodes, dimensionState):
        
        # Find the covariance for each cluster
        nodesMean = np.zeros((self.num_clusters, dimensionState))
            
        # Looping over the number of clusters
        for i in range(self.num_clusters):
            # If there is no data in that node
            if len(datanodes[i]) == 0:
                nodesMean[i,:]      = np.zeros(dimensionState)
            # If there is data in the node
            else:        
                datanode_of_cluster = np.asarray(datanodes[i])
                print(datanode_of_cluster.shape)
                nodesMean[i,:]      = np.mean(datanode_of_cluster, axis = 0)
                
        self.nodesMean = nodesMean
        
        return
    
    # Initialize the clustering knowing which are the datapoint and which
    # are the cluster assignments of the datapoints, i.e.,:
    # - data
    # - clustersSequence
    def LoadGraphFromDataAndAssignments(self, data, clustersSequence):
        
        self.num_clusters     = int(max(clustersSequence) - min(clustersSequence)) + 1
        
        firstClusterIndex     = np.min(clustersSequence)
        clustersSequence      = clustersSequence - firstClusterIndex
        clustersSequence      = clustersSequence.astype(int)
        self.clustersSequence = np.squeeze(clustersSequence)
        
        dimensionState        = data.shape[1]
        
        # Define one array containing the data for each cluster:
        datanodes = self.FindDataForEachCluster(data)  
        self.datanodes = datanodes
        
        # Find the mean of each cluster
        self.FindMeanForEachCluster(datanodes, dimensionState)
        
        # Find the covariance for each cluster
        self.FindCovarianceForEachCluster(datanodes, dimensionState)
        
        # Find the transition matrices
        self.CalculateAllTransitionMatrices()
        
        return

    
    # Function to save a graph as a matlab structure
    def SaveGraphToMATLAB(self, fileName):
        
        graphToSave = {
                'num_clusters'          : self.num_clusters,
                'nodesMean'             : self.nodesMean,
                'nodesCov'              : self.nodesCov,
                'transitionMat'         : self.transitionMat,
                'transMatsTime'         : self.transMatsTime,
                'windowedtransMatsTime' : self.windowedtransMatsTime,
                'clustersSequence'      : self.clustersSequence,
                'maxClustersTime'       : self.maxClustersTime
                }
        
        savemat(fileName, {'graph': graphToSave})
        
        return
        
        
    
    # A PRIVATE function to smoothen the cluster assignments.
    # Don't use this outside, as otherwise cluster assignment property and
    # transition matrices properties will not be coherent w.r.t. each other 
    # any more!!
    def __SmoothClusterAssignments(self, deleteThreshold = 2):
        
        # Bring to numpy to execute calculation
        flagTorch = False
        if type(self.clustersSequence) == torch.Tensor: 
            flagTorch             = True
            self.clustersSequence = self.clustersSequence.detach().cpu().numpy()   
            
        # Create array with same elements as original one
        clustersSequenceSmoothed  = copy.deepcopy(self.clustersSequence)
        
        # Number of data points in training
        dataLength = len(clustersSequenceSmoothed)
        
        # Looping over the data points
        for i in range(deleteThreshold, dataLength - (2*deleteThreshold)):

            currentCluster = self.clustersSequence[i]
            
            # Define window for cluster substitution
            windowBegin = i - deleteThreshold
            windowEnd   = i + 2*deleteThreshold - 1
            
            # Counting how many times the cluster appears in the window
            howManyTimesClusterAppearsInTheWindow = np.sum(self.clustersSequence[windowBegin:windowEnd] == currentCluster)
            
            if howManyTimesClusterAppearsInTheWindow > deleteThreshold:
                # If there are at least a number n = delete_threshold of that same
                # cluster assignment in the window, 
                # do nothing.
                dummy = 0
            else:
                # Find the other cluster that is present
                otherClustersIndex   = np.nonzero(self.clustersSequence[windowBegin:windowEnd] != currentCluster)[0] 
                
                # Try with each cluster appearing, to see if it appears more than the original one
                # and more than the other ones
                
                # As first most-appearing cluster, we take the original one
                frequencyOfClusterAppearingMost = howManyTimesClusterAppearsInTheWindow
                clusterAppearingMost            = currentCluster
                
                for j in range(len(otherClustersIndex)):
                    
                    # Cluster
                    otherCluster = self.clustersSequence[windowBegin-1+otherClustersIndex[j]]
                    # How many times it appears in the window
                    howManyTimesClusterAppearsInTheWindow = np.sum(self.clustersSequence[windowBegin:windowEnd] == otherCluster)
                    
                    # Is it more frequent?
                    if howManyTimesClusterAppearsInTheWindow > frequencyOfClusterAppearingMost:
                        frequencyOfClusterAppearingMost = howManyTimesClusterAppearsInTheWindow
                        clusterAppearingMost            = otherCluster
                        
                    # Substitute cluster
                    clustersSequenceSmoothed[i] = clusterAppearingMost
        
        # Bring back to torch if data was in torch
        if flagTorch == True:
            clustersSequenceSmoothed = torch.from_numpy(clustersSequenceSmoothed).float()
                    
        # Save sequence of data
        self.clustersSequence = clustersSequenceSmoothed
  
        return
    
    @staticmethod
    # Function to eliminate a row and column from a transition matrix
    def EliminateRowAndColumnFromMatrix(transitionMatrix, indexOfRowAndColumn):
        
        if type(transitionMatrix)   == np.ndarray:
            transitionMatrix = np.delete(transitionMatrix, indexOfRowAndColumn, axis=0)
            transitionMatrix = np.delete(transitionMatrix, indexOfRowAndColumn, axis=1)           
        elif type(transitionMatrix) == torch.Tensor:
            allIndicesExceptOneToEliminate = torch.arange(transitionMatrix.size(0))!=indexOfRowAndColumn
            transitionMatrix = transitionMatrix[allIndicesExceptOneToEliminate, :] 
            transitionMatrix = transitionMatrix[:, allIndicesExceptOneToEliminate] 
        
        return transitionMatrix
    
    @staticmethod
    # Function to eliminate a row from a nodesMean matrix
    def EliminateRowFromMatrix(nodesMean, indexOfRow):
        
        if type(nodesMean)   == np.ndarray:
            nodesMean = np.delete(nodesMean, indexOfRow, axis=0)  
        elif type(nodesMean) == torch.Tensor:
            allIndicesExceptOneToEliminate = torch.arange(nodesMean.size(0))!=indexOfRow
            nodesMean = nodesMean[allIndicesExceptOneToEliminate, :] 
        
        return nodesMean
    
    @staticmethod
    # Function to eliminate a value from a nodesMean matrix
    def EliminateValueFromVector(vector, index):
        
        if type(vector)   == np.ndarray:
            vector = np.delete(vector, index, axis=0)  
        elif type(vector) == torch.Tensor:
            allIndicesExceptOneToEliminate = torch.arange(vector.size(0))!=index
            vector = vector[allIndicesExceptOneToEliminate] 
        
        return vector
    
    # Function to eliminate a value from a clusters sequence
    def EliminateClusterFromClustersSequence(self, index):
        
        for i in range(len(self.clustersSequence)):
            
            currentCluster = self.clustersSequence[i] 
            
            if currentCluster >= index:
                self.clustersSequence[i] = currentCluster - 1
        
        return
    
    # If we want to eliminate one cluster from the graph
    # (for example, after smoothing, there is a cluster that is not used
    # any more)
    def EliminateOneCluster(self, indexOfCluster):
        
        # Number of clusters
        self.num_clusters    = self.num_clusters - 1
        
        # From NodesMean
        self.nodesMean       = ClusteringGraph.EliminateRowFromMatrix(self.nodesMean, indexOfCluster)
        # From NodesCov
        self.nodesCov.pop(indexOfCluster)
        # From Transition matrix
        self.transitionMat   = ClusteringGraph.EliminateRowAndColumnFromMatrix(self.transitionMat, indexOfCluster)
        
        # From temporal matrices - related values
        newMaxClustersTime = ClusteringGraph.EliminateValueFromVector(self.maxClustersTime, indexOfCluster)
        maxTimeOverall     = np.max(newMaxClustersTime)
        
        # Temporal transition matrices
        for i in range(int(np.max(self.maxClustersTime))):
            if i <= maxTimeOverall:
                self.transMatsTime[i] = ClusteringGraph.EliminateRowAndColumnFromMatrix(self.transMatsTime[i], indexOfCluster)
            else:
                self.transMatsTime.pop(i)
        
        # From temporal matrices - related values
        self.maxClustersTime = newMaxClustersTime
        
        # From clusters assignment
        self.EliminateClusterFromClustersSequence(indexOfCluster)
        
        #######################################################################
        # Additional features to the base one      
        if self.nodesCovPred != None:
            self.nodesCovPred.pop(indexOfCluster)
        if self.nodesCovD != None:
            self.nodesCovD.pop(indexOfCluster)
           
        return
    
    # Smooths the cluster assignments and recalculates all the properties consequently
    def SmoothClusterAssignmentsAndRecalculateProperties(self, deleteThreshold = 2):
        
        # Smoothen cluster assignments
        self.__SmoothClusterAssignments(deleteThreshold)
        # Recalculate ALL the transition matrices
        self.CalculateAllTransitionMatrices()
        
        # Are there any nan values in transition matrix? i.e., clusters that
        # were eliminated?
        countOverClusters = 0
        for i in range(self.num_clusters):
            if type(self.transitionMat)   == np.ndarray:
                if np.isnan(self.transitionMat[countOverClusters,0]):
                    print(countOverClusters)
                    self.EliminateOneCluster(countOverClusters)
                    countOverClusters -= 1
            elif type(self.transitionMat) == torch.Tensor:
                if torch.isnan(self.transitionMat[countOverClusters,0]):
                    self.EliminateOneCluster(countOverClusters)
                    countOverClusters -= 1
            countOverClusters += 1
        
        return
    
    # Function to create the graph interactions 
    # (To use in Graph Matching code).
    # Needs as input self.clustersSequence.
    # Outputs: - index: key of clusters
    #          - interaction: list of transitions between clusters
    def FindInteractionsList(self):
        
        # Create the index list: it just gives a correspondence key-index
        index = dict()
        for i in range (0, self.num_clusters):
            index[i] = i
           
        # list with all the interactions from a node i to a node j, with i != j
        interactions = []
        for i in range (0, len(self.clustersSequence) - 1):
            # If we have a movement from a cluster to another one
            #if idx[i + 1] != idx[i]:
            # Add the interaction to the list
            pair = [self.clustersSequence[i + 1], self.clustersSequence[i]]
            interactions.append(pair)
        # Sort the list in ascending order of numbers
        interactions.sort() 
        
        self.indicesList      = index
        self.interactionsList = interactions
        
        return
    
    def FindTransitionMatrix(self):
        
        # Empty transition matrix
        transitionMat = np.zeros((self.num_clusters,self.num_clusters));
        
        #Total length of training data
        trainingDataLength = len(self.clustersSequence)
        
        # Count transitions from one cluster to another one
        for i in range(trainingDataLength - 1):
            transitionMat[self.clustersSequence[i],self.clustersSequence[i+1]] += 1;
            
        # Normalize transition matrix
        for i in range(self.num_clusters):
            transitionMat[i,:] = transitionMat[i,:]/np.sum(transitionMat[i,:])
        
        self.transitionMat = transitionMat;
        
        return
    
    def BringGraphToTorch(self):        
        self.nodesMean            = torch.from_numpy(self.nodesMean).to(device)
        self.clustersSequence     = torch.from_numpy(self.clustersSequence).to(device)
        for i in range(self.num_clusters):
            self.nodesCov[i]      = torch.from_numpy(self.nodesCov[i]).to(device)
        for i in range(len(self.transMatsTime)):
            self.transMatsTime[i] = torch.from_numpy(self.transMatsTime[i]).to(device)
        for i in range(len(self.transMatsTime)):
            self.windowedtransMatsTime[i] = torch.from_numpy(self.windowedtransMatsTime[i]).to(device)
            
        if hasattr(self, 'additionalClusterInfo'):
            self.additionalClusterInfo = torch.from_numpy(self.additionalClusterInfo).to(device)

        self.BringTransitionMatrixToTorch()
        
        #######################################################################
        # Additional features to the base one      
        if self.nodesCovPred != None:
            for i in range(self.num_clusters):
                self.nodesCovPred[i]   = torch.from_numpy(self.nodesCovPred[i]).to(device)
        if self.nodesCovD != None:
            for i in range(self.num_clusters):
                self.nodesCovD[i]      = torch.from_numpy(self.nodesCovD[i]).to(device)
        
        return
        
    
    def BringTransitionMatrixToTorch(self):
        self.transitionMat = torch.from_numpy(self.transitionMat).to(device)
        return
    
    # Print function
    def Print(self):
        
        print('Number of clusters in graph:')
        print(self.num_clusters)
        print('Sequence of clusters')
        print(self.clustersSequence)
        #print('Cluster means in graph:')
        #print(self.nodesMean)
        #print('Synchronized data in graph:')
        #print(self.data_m)
        #print('Cluster covariances in graph:')
        #print(self.nodesCov)
        
        return
            
    # Find the distance of a set of points from each node of a cluster graph
    # inputs: - points_sequence: sequence of odometry/distance points
    #                            The form of this data should be:
    #                            > dimension 0: number of sequences;
    #                            > dimension 1: length of sequence (fixed len);
    #                            > dimension 2: dimension of sequence (e.g., if pos x, pos y = 2).
    #         - obsCovariance: covariance of observation
    # output: distances of each datapoint in the sequences from each cluster. 
    #                            The form of this data will be:
    #                            > dimension 0: number of sequences;
    #                            > dimension 1: length of sequence (fixed len);
    #                            > dimension 2: number of clusters (num_clusters).
    def FindBhattaDistancesFromSequencesToClusters(self, points_sequence, obsVariance):
        
        # Information about the sequence of points
        num_sequences        = points_sequence.shape[0] # how many sequences
        num_time_in_sequence = points_sequence.shape[1] # length of sequence
        
        # Calculating the distances
        if type(points_sequence)   == np.ndarray:
            distances     = np.zeros((num_sequences,num_time_in_sequence,self.num_clusters))
            obsCovariance = np.diag(obsVariance)
        elif type(points_sequence) == torch.Tensor:
            distances     = torch.zeros(num_sequences,num_time_in_sequence,self.num_clusters).to(device)
            obsCovariance = torch.diag(obsVariance).to(device)
        
        # looping over the number of sequences
        for i in range(num_sequences):
            # Looping over the number of elements in a sequence
            for j in range(num_time_in_sequence):
                # current obs
                measurement = points_sequence[i,j,:]
                
                # Calculating Bhattacharya distance between observation and clusters
                for index_s in range(self.num_clusters):
                    
                    if type(points_sequence)   == np.ndarray:
                        nodesCovDiagCurrCluster = np.diag(self.nodesCov[index_s])
                        currDistance = d_utils.CalculateBhattacharyyaDistance(measurement, 
                                                                          obsCovariance, 
                                                                          self.nodesMean[index_s,:], 
                                                                          nodesCovDiagCurrCluster)

                    elif type(points_sequence) == torch.Tensor:
                        
                        nodesCovDiagCurrCluster = torch.diag(self.nodesCov[index_s]).to(device)
                        currDistance = d_utils.CalculateBhattacharyyaDistanceTorch(measurement, 
                                                                          obsCovariance, 
                                                                          self.nodesMean[index_s,:], 
                                                                          nodesCovDiagCurrCluster)
                        
                    distances[i,j,index_s] = currDistance
                    
        return distances
    
    # Different to the previous one: we suppose to have a single sequence of length 1, so there
    # is no need for the two 'for' loops.
    def FindBhattaDistancesFromSequencesToClustersSinglePoint(self, points_sequence, obsVariance):
        
        # Information about the sequence of points
        num_sequences        = points_sequence.shape[0] # how many sequences
        num_time_in_sequence = points_sequence.shape[1] # length of sequence
        
        # Calculating the distances
        if type(points_sequence)   == np.ndarray:
            distances     = np.zeros((num_sequences,num_time_in_sequence,self.num_clusters))
            obsCovariance = np.diag(obsVariance)
        elif type(points_sequence) == torch.Tensor:
            distances     = torch.zeros(num_sequences,num_time_in_sequence,self.num_clusters).to(device)
            obsCovariance = torch.diag(obsVariance).to(device)

        # current obs
        measurement = points_sequence[0,0,:]
        
        # Calculating Bhattacharya distance between observation and clusters
        for index_s in range(self.num_clusters):
            
            if type(points_sequence)   == np.ndarray:
                nodesCovDiagCurrCluster = np.diag(self.nodesCov[index_s])
                currDistance = d_utils.CalculateBhattacharyyaDistance(measurement, 
                                                                  obsCovariance, 
                                                                  self.nodesMean[index_s,:], 
                                                                  nodesCovDiagCurrCluster)

            elif type(points_sequence) == torch.Tensor:
                
                nodesCovDiagCurrCluster = torch.diag(self.nodesCov[index_s]).to(device)
                currDistance = d_utils.CalculateBhattacharyyaDistanceTorch(measurement, 
                                                                  obsCovariance, 
                                                                  self.nodesMean[index_s,:], 
                                                                  nodesCovDiagCurrCluster)
                
            distances[0,0,index_s] = currDistance
                    
        return distances
    
    '''
    def FindBhattaDistancesFromSequencesToClusters(self, points_sequence, obsVariance):
        
        # Information about the sequence of points
        num_sequences        = points_sequence.shape[0] # how many sequences
        num_time_in_sequence = points_sequence.shape[1] # length of sequence
        
        # Calculating the distances
        if type(points_sequence)   == np.ndarray:
            distances     = np.zeros((num_sequences,num_time_in_sequence,self.num_clusters))
            obsCovariance = np.diag(obsVariance)
        elif type(points_sequence) == torch.Tensor:
            distances     = torch.zeros(num_sequences,num_time_in_sequence,self.num_clusters).to(device)
            obsCovariance = obsVariance
        
        # looping over the number of sequences
        for i in range(num_sequences):
            # Looping over the number of elements in a sequence
            for j in range(num_time_in_sequence):
                # current obs
                obs = points_sequence[i,j,:]
                
                # Calculating Bhattacharya distance between observation and clusters
                for index_s in range(self.num_clusters):
                    
                    if type(points_sequence)   == np.ndarray:
                        nodesCovDiagCurrCluster = np.diag(self.nodesCov[index_s])
                        currDistance = d_utils.CalculateBhattacharyyaDistance(obs, 
                                                                          obsCovariance, 
                                                                          self.nodesMean[index_s,:], 
                                                                          nodesCovDiagCurrCluster)

                    elif type(points_sequence) == torch.Tensor:
                        
                        nodesCovDiagCurrCluster = self.nodesCov[index_s].to(device)
                        currDistance = d_utils.CalculateBhattacharyyaDistanceTorch(obs, 
                                                                          obsCovariance, 
                                                                          self.nodesMean[index_s,:], 
                                                                          nodesCovDiagCurrCluster)

                    distances[i,j,index_s] = currDistance
                    
        return distances
    '''