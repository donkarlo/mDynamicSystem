
# GROWING NEURAL GAS ##########################################################


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.manifold import TSNE

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import mat4py # for loading from matlab

import scipy # for pairwise distance
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

import PyQt5

###############################################################################

# To use for TSNE plotting in case of high-dimensional data
def reduce_dimensionality(var, perplexity=10):
    
    dim = var.shape[-1]
        
    if(dim>2):
        tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=1000)
        var_2d = tsne.fit_transform(var)
    else:
        var_2d = np.asarray(var)
            
    return var_2d
            

# LOADING THE DATA TO CLUSTER #################################################


# Positional data
inputData = 'manip-gps-3-d-pos-vel.txt'


file = open(inputData, "r")

inputData = []

for lineCounter, curLine in enumerate(file):
    
    curLineSeparated = curLine.split(',');
    
    pos_x = float(curLineSeparated[0])
    pos_y = float(curLineSeparated[1])
    pos_z = float(curLineSeparated[2])
    vel_x = float(curLineSeparated[3])
    vel_y = float(curLineSeparated[4])
    vel_z = float(curLineSeparated[5])
    
    current_data_line = np.zeros(6)
    
    current_data_line[0] = pos_x
    current_data_line[1] = pos_y
    current_data_line[2] = pos_z
    current_data_line[3] = vel_x
    current_data_line[4] = vel_y
    current_data_line[5] = vel_z
    
    inputData.append(current_data_line)
    
inputData = np.asarray(inputData)



# PARAMETERS DEFINITION #######################################################

maxNodes = 20;                                        # Number of nodes
MaxIt = 10;                                     # Iteration (repetition of input data)
L_growing = 1000;                               # Growing rate
epsilon_b = 0.05;                               # Movement of winner node
epsilon_n = 0.0006;                             # Movement of all other nodes except winner
alpha = 0.5;                                    # Decaying global error and mUtility
delta = 0.9995;                                 # Decaying local error and mUtility
T = 100;                                        # It could be a function of params.L_growing, e.g., params.LDecay = 2*params.L_growing
L_decay = 1000;                                 # Decay rate should be faster than the growing then it will remove extra nodes
alpha_utility = 0.0005;     
k = 1.5;
seedvector = 2;


PlotFlag = True

# CLUSTERING PROCEDURE ########################################################

# Normalization of the input data
minDataNorm = np.min(inputData, axis = 0)
dataNorm = inputData - np.tile(minDataNorm, (inputData.shape[0] , 1)) # tile(a, (m, n)) = repmat(a, m, n)
maxDataNorm = np.max(dataNorm, axis = 0)
CV = dataNorm/np.tile(maxDataNorm, (inputData.shape[0] , 1))
inputNormOrd = CV

# Size of input data (number of training samples)
nData = CV.shape[0]  
#Dimension of input data                                                  
nDim = CV.shape[1]                                                       

# Permutation of the rows of the input data vector
permutationValues = np.random.RandomState(seed=seedvector).permutation(nData)
CV = CV[permutationValues, :]

CVmin = np.min(CV, axis = 0)
CVmax = np.max(CV, axis = 0)

# Initialization ##############################################################
# Initial 2 nodes for training the algorithm
Ni = 2                                                                    

wNorm = np.zeros((Ni, nDim))
# It returns an array of random numbers generated from the continuous uniform 
# distributions with lower and upper endpoints specified by 'CVmin' and 'CVmax'.
for i in range(Ni):
    wNorm[i,:] = np.random.uniform(low=CVmin, high=CVmax, size=None)                                        

# error
E = np.zeros(Ni)
# mUtility
utility = np.ones(Ni)
# Connections between nodes
C = np.zeros((Ni, Ni))
# Ages of the edges
t = np.zeros((Ni, Ni))


# Loop ########################################################################

nx = 0

# Loop over the number of iterations parameter
for it in range(MaxIt):
    
    print("Iteration " + str(it) + " out of " + str(MaxIt))
    print("Number of nodes in current iteration: " + str(wNorm.shape[0]))
    
    # Loop over the number of data
    for c in range (nData):
        
        # Select Input
        
        # Counter of cycles inside the algorithm
        nx = nx + 1
        # pick first input vector from permuted inputs
        x = CV[c, :]
        
        # COMPETITION AND RANKING
        # pairwise distance between normalized input value and the normalized node means
        # X = np.tile(x, (wNorm.shape[0],1))
        X = np.expand_dims(x, axis=0)
        X_state = X[:, 0: int(nDim/2)]
        X_deriv = X[:, int(nDim/2):nDim]
        wNorm_state = wNorm[:, 0: int(nDim/2)]
        wNorm_deriv = wNorm[:, int(nDim/2):nDim]
        d_state = metrics.pairwise_distances(X=X_state, Y=wNorm_state, metric='euclidean')[0][:]
        d_deriv = metrics.pairwise_distances(X=X_deriv, Y=wNorm_deriv, metric='euclidean')[0][:]
        
        value_state = d_state/np.sum(d_state)
        value_deriv = d_deriv/np.sum(d_deriv)
        
        d = value_state + value_deriv
        
        # Organize distances between nodes and the first data point in an ascending order
        SortOrder = np.argsort(d)
        
        # Closest node index to the first data point
        s1 = SortOrder[0] 
        # Second closest node index to the first data point                                              
        s2 = SortOrder[1]
        
        # AGING
        # Increment the age of all edges emanating from s1
        t[s1, :] = t[s1, :] + 1                                             
        t[:, s1] = t[:, s1] + 1
        
        # Add Error
        dist0  = np.power(d[s1],2)
        dist1  = np.power(d[s2],2)
        E[s1] = E[s1] + dist0
        
        # Utility
        # Initial mUtility is zero in first case and dist is the error of first node
        deltaUtility =  dist1 - dist0        
        # Difference between error of two nodes
        utility[s1] =  utility[s1] + deltaUtility
        
        # ADAPTATION
        # Move the nearest distance node and it's neibors towards the input signal 
        # by fractions Eb and En resp.
        # 1) move nearest node
        wNorm[s1,:] = wNorm[s1,:] + epsilon_b*(x-wNorm[s1,:])
        
        # Take all the connections of the closest node to the data in question
        Ns1 = np.where(C[s1,:] == 1)
        # 2) move neighbors
        #for j in Ns1[0]:
        #    wNorm[j,:] = wNorm[j,:] + epsilon_n*(x-wNorm[j,:])   
        wNorm[Ns1, :] = wNorm[Ns1,:] + epsilon_n*(x-wNorm[Ns1,:])
        
        # Create link
        # If s1 and s2 are connected by an edge , set the age of this edge to 
        # zero , it such edge doesn't exist create it
        C[s1,s2] = 1                                               
        C[s2,s1] = 1
        # Age of the edge
        t[s1,s2] = 0                                                     
        t[s2,s1] = 0
        
        # Remove old links
        # remove edges with an age larger than Amax(a threshold value)
        C[t > T] = 0       
        
        # Number of connections of each node                                                
        nNeighbor = np.sum(C, axis = 1)
        # Eliminate alone nodes from the C and t matrix and 
        # from wNorm, E and mUtility vector
        # AloneNodes = (nNeighbor==0)
        indexAloneNodes = np.where(nNeighbor == 0)
        C = np.delete(C, (indexAloneNodes), axis=0)
        C = np.delete(C, (indexAloneNodes), axis=1)
        t = np.delete(t, (indexAloneNodes), axis=0)
        t = np.delete(t, (indexAloneNodes), axis=1)
        
        wNorm = np.delete(wNorm, (indexAloneNodes), axis=0)
        E = np.delete(E, (indexAloneNodes), axis=0)
        utility = np.delete(utility, (indexAloneNodes), axis=0)
        
        # ADD NEW NODES at every L_growing
        
        if ((np.remainder(nx, L_growing) == 0) and (wNorm.shape[0] < maxNodes)):
            
            # Determine the unit q with the maximum accumulated error
            q = np.argmax(E)
            # Maximum index related to the error related to a connected node
            f = np.argmax(C[:,q]*E)
            
            # Total number of nodes
            r = wNorm.shape[0] + 1
            index_r = r-1 # to index the new node
            # Insert a new unit r halfway between q and it's neibor f with 
            # the largest error variable
            newNode = (wNorm[q,:] + wNorm[f,:])/2
            wNorm = np.vstack([wNorm, newNode])
            
            # Adding one row and column to C and t
            C_temp = np.zeros((r, r))
            C_temp[0:r-1, 0: r-1] = C
            C = C_temp
            
            t_temp = np.zeros((r, r))
            t_temp[0:r-1, 0: r-1] = t
            t = t_temp
            
            # Remove old connections and introduce the presence of the
            # new created node
            C[q,f] = 0; # eliminating connections between the two former neighbors
            C[f,q] = 0;
            C[q,index_r] = 1; # Creating connections between old nodes and new one
            C[index_r,q] = 1;
            C[index_r,f] = 1;
            C[f,index_r] = 1;
            t[index_r,:] = 0;
            t[:, index_r] = 0;
            
            # Decrease the error variable of q and f by multiplying them with a constant 'alpha'
            E[q] = alpha*E[q]
            E[f] = alpha*E[f]
            # Initialize the error of the new node equal to error of the winner node
            newError = E[q]
            E = np.append(E,newError)
            # Decrease the error variable of q and f by multiplying them with a constand 'alpha'
            #mUtility[q] = alpha*mUtility[q]
            #mUtility[f] = alpha*mUtility[f]
            newUtility = 0.5 *( utility[q] + utility[f] )
            utility = np.append(utility,newUtility)
        
        # REMOVE NODES at every L_decay
        
        if (np.remainder(nx, L_decay) == 0):
            
            # Maximum accumulated error
            max_E = np.max(E)
            # Node node_useless having minimum mUtility
            min_utility = np.min(utility)
            node_useless = np.argmin(utility)
            # Utility factor
            CONST = min_utility * k
            
            if (CONST < max_E):
                # Remove the connection having smaller mUtility factor
                C = np.delete(C, (node_useless), axis=0)
                C = np.delete(C, (node_useless), axis=1)
                # Remove the node having smaller mUtility factor
                wNorm = np.delete(wNorm, (node_useless), axis=0)
                # Remove the min mUtility value from the mUtility vector
                utility = np.delete(utility, (node_useless), axis=0)  
                # Remove error vector correspond to the node having min mUtility
                E = np.delete(E, (node_useless), axis=0)  
                # Remove aging vector correspond to the node having min mUtility
                t = np.delete(t, (node_useless), axis=0)
                t = np.delete(t, (node_useless), axis=1)
                
            #E = alpha*E
            #mUtility = alpha*mUtility
                
        # Decrease errors
        # Decrease error variables by multiplying them with a constant delta
        E = delta * E
        # Decrease the mUtility by alpha_utility constant
        utility = delta * utility
        
        
    if PlotFlag == True: # to ADD
        
        print("plotting...")
    
###############################################################################
        
# Clusters of input
dataColorNode = np.zeros(nData)
for c in range(nData):
    x = inputNormOrd[c,:]
    X = np.expand_dims(x, axis=0)
    d = metrics.pairwise_distances(X=X, Y=wNorm, metric='euclidean')[0][:]
    minNode = np.argmin(d)
    
    dataColorNode[c] = minNode
       
###############################################################################
# PLOTTING FINAL

# Positional case
colors = {0:'black', 1:'grey', 2:'blue', 3:'cyan', 4:'lime', 5:'green', 6:'yellow', 7:'gold', 8:'red', 9:'maroon'}
        
plt.ion()
plt.show()
                
f, axarr = plt.subplots(1, 1, figsize=(10, 10))
var_2d = inputNormOrd[:, 0:2]
                
for number, color in colors.items():
    axarr.scatter(x=var_2d[dataColorNode==number, 0], y=var_2d[dataColorNode==number, 1], color=color, label=str(number))
    axarr.legend()
           
axarr.grid()
plt.draw()
#plt.pause(0.002)
f.suptitle("Plotting clustering", fontsize=20)


import ClusteringGraph as CG

clusterGraph = CG.ClusteringGraph(inputData.shape[1])
clusterGraph.LoadGraphFromDataAndAssignments(inputData, dataColorNode)
clusterGraph.SaveGraphToMATLAB('graph.mat')



###############################################################################
###############################################################################
###############################################################################
############################## MJPF ###########################################
###############################################################################
###############################################################################
###############################################################################



###############################################################################
# Bring the MJPF to python

###############################################################################
# Import of all necessary functions

import numpy as np
from numpy.core.records import fromarrays
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import random
import mat4py # this is for saving to MATLAB
import os
import warnings

###############################################################################
# Custom codes
import DefineColors 
import LoadData as LD
import BuildVocabulary as BV
import Config
import KF
import DistanceCalculations as DC

###############################################################################
# This is to say if you want the plots inline or on a separate window

from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')

###############################################################################

warnings.filterwarnings("ignore")

###############################################################################
# Colors for clustering plotting
colors_array = DefineColors.DefineColors()

###############################################################################
# ---------------------------- USER SETTINGS ----------------------------------
###############################################################################

path = os.path.dirname(os.path.abspath(__file__)) 

# Configuring the settings
config = Config.ConfigureSettings() # Insert your settings in this file

###############################################################################
# ---------------------------- DATA LOADING  ----------------------------------
###############################################################################

# Loading the VOCABULARY

nClusters                = clusterGraph.num_clusters
nodesMean                = clusterGraph.nodesMean
nodesCov                 = clusterGraph.nodesCov
dataColorNode            = clusterGraph.clustersSequence
transitionMat            = clusterGraph.transitionMat
temporalTransitionMatrix = clusterGraph.windowedtransMatsTime
maxClustersTime          = clusterGraph.maxClustersTime


# Positional data
inputData = 'manip-gps-3-d-pos-vel-turn-left.txt'


file = open(inputData, "r")

inputData = []

for lineCounter, curLine in enumerate(file):
    
    curLineSeparated = curLine.split(',');
    
    pos_x = float(curLineSeparated[0])
    pos_y = float(curLineSeparated[1])
    pos_z = float(curLineSeparated[2])
    vel_x = float(curLineSeparated[3])
    vel_y = float(curLineSeparated[4])
    vel_z = float(curLineSeparated[5])
    
    current_data_line = np.zeros(6)
    
    current_data_line[0] = pos_x
    current_data_line[1] = pos_y
    current_data_line[2] = pos_z
    current_data_line[3] = vel_x
    current_data_line[4] = vel_y
    current_data_line[5] = vel_z
    
    inputData.append(current_data_line)
    
inputData = np.asarray(inputData)

dataLength = 1000


data = inputData
GS   = inputData
    
###############################################################################
# Parameters Used in the Filtering Process

# Length of testing data
dataLength   = data.shape[0]
# Number of latent states
GSVDimension = data.shape[1]
# Number of clusters
nSuperStates = nClusters

# Number of particles 
maxNodes = config['nParticles'];

skewValue = config['skewValue']

###############################################################################
# Transition matrix
A = np.eye(GSVDimension, GSVDimension)
# Measurement matrix
H = np.eye(GSVDimension, GSVDimension)
# Control input
B = np.zeros((GSVDimension, int(GSVDimension/2)))
B[0,0] = 1
B[1,1] = 1
B[2,0] = 1
B[3,1] = 1

###############################################################################
# Generate Observation Noise (Observation Noise): 
# v ~ N(0,R) meaning v is gaussian noise with covariance R
Var_ONoise = 1e-2                    # Observation Noise variance % it was = 2
Mu_ONoise  = 0                       # Observation Noise mean
Std_ONoise = np.sqrt(Var_ONoise)     # Standard deviation of the observation noise

###############################################################################
# Empty values to fill
predicted_superstate = np.zeros((maxNodes, dataLength));
predicted_state      = np.zeros((GSVDimension, dataLength, maxNodes));
predicted_cov_state  = np.zeros((GSVDimension, GSVDimension, dataLength, maxNodes));
updated_state        = np.zeros((GSVDimension, dataLength, maxNodes));
updated_cov_state    = np.zeros((GSVDimension, GSVDimension, dataLength, maxNodes));

predicted_state_resampled      = np.zeros((GSVDimension, dataLength, maxNodes))
predicted_superstate_resampled = np.zeros((maxNodes, dataLength))
updated_state_resampled        = np.zeros((GSVDimension, dataLength, maxNodes))
updated_cov_state_resampled    = np.zeros((GSVDimension, GSVDimension, dataLength, maxNodes))
CLA_resampled                  = np.zeros((maxNodes, dataLength))
CLB_resampled                  = np.zeros((maxNodes, dataLength))

w            = np.zeros((maxNodes, 1))
weightscoeff = np.zeros((maxNodes, dataLength))
t            = np.zeros((maxNodes, 1))

probability_lamdaS    = np.zeros((dataLength, nSuperStates))
predicted_superstates = np.zeros((maxNodes, dataLength))

# Anomalies
CLB = np.zeros((maxNodes, dataLength))
KLDabn_all = np.zeros((dataLength,1))

histogram_before_update = np.zeros((nSuperStates, dataLength))
histogram_after_update  = np.zeros((nSuperStates, dataLength))

discreteEvents_basedOn_LamdaS = np.zeros((dataLength, 1))

min_innovation = np.zeros((dataLength,1))
minCLAs        = np.zeros((dataLength,1))
minCLBs        = np.zeros((dataLength,1))

###############################################################################
# MAIN LOOP

plt.ion()
#fig, axes = plt.subplots(1, 1, figsize=(10, 5))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 10))
fig.suptitle('Positions and anomalies')

# min and max values of position for position plotting
max_pos_x    = np.max(data[:, 0])
max_pos_y    = np.max(data[:, 1])
min_pos_x    = np.min(data[:, 0])
min_pos_y    = np.min(data[:, 1])


for i in range(dataLength-1) :
    
    if config['printCounter'] == True:
        print (str(i) + " out of " + str(dataLength))
        
    # Gaussian Observation Noise
    ONoise = Std_ONoise * np.random.randn(GSVDimension, 1) + Mu_ONoise*np.ones((GSVDimension, 1))
    OVar   = np.var(ONoise)  
    R      = np.eye(GSVDimension)*OVar
    
    # ------ INITIAL STEP ------ > #
    if i == 0: # Initial step
        for n in range(maxNodes):
            
            predicted_state[:,i,n] = np.random.multivariate_normal(data[i,:], R)
            predicted_cov_state_initial = R;
            t[n] = 1
            weightscoeff[n,i] = 1 / maxNodes
            
            # Observe first Measurement Zt
            current_measurement = data[i,:]
            
            # Calculate Message Back-ward Propagated towards discrete-level (S)
            if n == 0:
                probability_lamdaS[i, :] = DC.calculateLamdaS(nSuperStates, current_measurement, nodesMean, R, nodesCov, skewValue)
                    
            predicted_superstate       = random.choices(np.arange(nSuperStates), weights = probability_lamdaS[i, :], k = 1)[0]
            predicted_superstates[n,i] = predicted_superstate

            ## ------------------------ UPDATE STEP ---------------------------
            # -- update states -- #
            ## during the update kalman computes the posterior p(x[k] | z[1:k]) = N(x[k] | m[k], P[k])
            updated_state[:,i,n], updated_cov_state[:,:,i,n] = KF.kf_update(predicted_state[:,i,n], 
                         predicted_cov_state_initial, current_measurement, H, R)
            
            ## ------------------------- ANOMALIES ----------------------------
            
            ## Calculate Abnormalities
            # -- continuous level -- #
            
            # measure bhattacharrya distance between p(xk/xk-1) and p(xk/sk)
            
            CLB[n,i] = DC.CalculateBhattacharyyaDistance(predicted_state[:,i,n]           , np.diag(predicted_cov_state_initial),
                                                         nodesMean[predicted_superstate,:], np.diag(nodesCov[predicted_superstate]))

            # -- update superstates -- #
            w[n] = weightscoeff[n,i]*probability_lamdaS[i,predicted_superstate]
              
        ## Calculate Histogram before update
        for ii in range(nSuperStates):
            elements = np.where(predicted_superstates[:,i] == ii)
            histogram_before_update[ii,i] = len(elements[0])
            
        ## ------------------------- PF Resampling ----------------------------
        w = w/np.sum(w); # normalize weights
        # multinomial distribution to pick multiple likely particles
        swap_index = random.choices(np.arange(maxNodes), weights = w, k = maxNodes)
        for n in range(maxNodes):
            predicted_state_resampled[:,i,n]     = predicted_state[:,i,swap_index[n]]
            predicted_superstate_resampled[n,i]  = predicted_superstates[swap_index[n],i]
            updated_state_resampled[:,i,n]       = updated_state[:,i,swap_index[n]]
            updated_cov_state_resampled[:,:,i,n] = updated_cov_state[:,:,i,swap_index[n]]
            CLB_resampled[n,i]                   = CLB[swap_index[n], i]
        predicted_state       = predicted_state_resampled
        predicted_superstates = predicted_superstate_resampled
        updated_state         = updated_state_resampled
        updated_cov_state     = updated_cov_state_resampled
        CLB                   = CLB_resampled
        ## Calculate Histogram after update
        for ii in range(nSuperStates):
            elements = np.where(predicted_superstates[:,i] == ii)
            histogram_after_update[ii,i] = len(elements[0])
            
        weightscoeff[:,i+1] = 1 / maxNodes
        
        ## Calculate Abnormalities
        # -- discrete level -- #
        KLDA = DC.KLD_Abnormality(nSuperStates, maxNodes, histogram_after_update[:, i], transitionMat, probability_lamdaS[i, :], KLDAbnMax=10000)
        KLDabn_all[i] = KLDA
        
        ## Calculate Generalized Errors
        indexMaxLamdaS = np.argmax(probability_lamdaS[i,:])
        discreteEvents_basedOn_LamdaS[i] = indexMaxLamdaS   
        
        # ------ INITIAL STEP ------ < #
        

    # ------ FOLLOWING STEPS ------ > #  
    if i > 0:
        
        for n in range(maxNodes):
            ## Discrete-Level prediction
            # Select row of transition matrix
            transitionMatRow = transitionMat[int(predicted_superstates[n,i-1]),:]
            # Considering time matrices, if we have been in a cluster for
            # more than one time instant
            maxTimeCurrentCluster = maxClustersTime[int(predicted_superstates[n,i-1])]
            
            if t[n][0] > 1 and t[n][0] < maxTimeCurrentCluster:
                # select the temporal transition matrix related to being
                # in the current cluster for t(n) instances
                curr_temporalTransitionMatrix = temporalTransitionMatrix[int(t[n])]
                temporalTransitionMatRow      = curr_temporalTransitionMatrix[int(predicted_superstates[n,i-1]),:]
                finalTransitionMatRow         = (temporalTransitionMatRow + transitionMatRow)/2
                finalTransitionMatRow         = finalTransitionMatRow/np.sum(finalTransitionMatRow)
                
            elif t[n][0] > 1 and t[n]>= maxTimeCurrentCluster:
                # select the last temporal transition matrix
                curr_temporalTransitionMatrix = temporalTransitionMatrix[int(maxTimeCurrentCluster)-1]
                temporalTransitionMatRow      = curr_temporalTransitionMatrix[int(predicted_superstates[n,i-1]),:]
            
                # If we have spent more time in the cluster than usual, the
                # probability of all clusters becomes more equal
                # This is to avoid getting stuck in a cluster
                probability_passage_to_all    = 1*np.abs(maxTimeCurrentCluster- t[n][0])/(maxNodes * maxTimeCurrentCluster)
                
                finalTransitionMatRow         = (temporalTransitionMatRow + transitionMatRow)/2 + probability_passage_to_all
                finalTransitionMatRow         = finalTransitionMatRow/np.sum(finalTransitionMatRow)
                
            else:
                finalTransitionMatRow = transitionMatRow
            
            # I find probability of next superstate
            probability = random.choices(np.arange(nSuperStates), weights = finalTransitionMatRow, k = 1)[0]
            predicted_superstates[n,i] = probability
            predicted_superstate       = int(predicted_superstates[n,i])
            
            # Increasing the time we have spent in the cluster (if we stayed in same cluster)
            if(predicted_superstates[n,i-1] == predicted_superstates[n,i]):
                t[n] = t[n] + 1                                           # If same superstate, add 1
            else:
                t[n] = 1                                                  # Else rinizialize by 1
                  
            ## Calculate Histogram before update
            for ii in range(nSuperStates):
                elements = np.where(predicted_superstates[:,i] == ii)
                histogram_before_update[ii,i] = len(elements[0])
                
            ## Continuous-Level prediction
            # Xt = AXt-1 + BUst-1 + wt
            currentState = updated_state[:,i-1,n]
            currentCov   = updated_cov_state[:,:,i-1,n]
            
            U  = nodesMean[int(predicted_superstates[n,i-1]),int((GSVDimension/2)):GSVDimension]
            Q2 = nodesCov[int(predicted_superstates[n,i-1])]
            
            [predicted_state[:,i,n], predicted_cov_state[:,:,i,n]] = KF.kf_predict(currentState, currentCov, A, Q2, B, U)
            
            ## Receive new Measurement Zt
            current_measurement = data[i,:]
            
            ## Calculate Message Back-ward Propagated towards discrete-level (S)
            if n == 0:
                probability_lamdaS[i, :] = DC.calculateLamdaS(nSuperStates, current_measurement, nodesMean, R, nodesCov, skewValue)
                
            ## Calculate Abnormalities
            # -- discrete level -- #
            KLDA = DC.KLD_Abnormality(int(nSuperStates), maxNodes, histogram_after_update[:, i - 1], transitionMat, probability_lamdaS[i, :], KLDAbnMax=10000)
            KLDabn_all[i] = KLDA
            # -- continuous level -- #
            
            # measure bhattacharrya distance between p(xk/xk-1) and p(xk/sk)
            
            CLB[n,i] = DC.CalculateBhattacharyyaDistance(predicted_state[:,i,n], 
                                                         np.diag(predicted_cov_state[:,:,i,n]),
                                                         nodesMean[int(predicted_superstates[n,i-1]),:], 
                                                         np.diag(nodesCov[int(predicted_superstates[n,i-1])]))
            
            
            ## UPDATE STEP
            # -- update states -- #
            ## during the update kalman computes the posterior p(x[k] | z[1:k]) = N(x[k] | m[k], P[k])
            updated_state[:,i,n], updated_cov_state[:,:,i,n] = KF.kf_update(predicted_state[:,i,n], 
                         predicted_cov_state[:,:,i,n], current_measurement, H, R)
            
            # -- update superstates -- #
            w[n] = weightscoeff[n,i]*probability_lamdaS[i,predicted_superstate]
            
        ## ------------------------- PF Resampling ----------------------------
        w = w/np.sum(w); # normalize weights
        # multinomial distribution to pick multiple likely particles
        swap_index = random.choices(np.arange(maxNodes), weights = w, k = maxNodes)
        for n in range(maxNodes):
            predicted_state_resampled[:,i,n]       = predicted_state[:,i,swap_index[n]]
            #predicted_cov_state_resampled[:,:,i,n] = predicted_cov_state[:,:,i,swap_index[n]]
            predicted_superstate_resampled[n,i]    = predicted_superstates[swap_index[n],i]
            updated_state_resampled[:,i,n]         = updated_state[:,i,swap_index[n]]
            updated_cov_state_resampled[:,:,i,n]   = updated_cov_state[:,:,i,swap_index[n]]
            CLB_resampled[n,i]                     = CLB[swap_index[n], i]
        predicted_state       = predicted_state_resampled
        #predicted_cov_state   = predicted_cov_state_resampled
        predicted_superstates = predicted_superstate_resampled
        updated_state         = updated_state_resampled
        updated_cov_state     = updated_cov_state_resampled
        CLB                   = CLB_resampled
        ## Calculate Histogram after update
        for ii in range(nSuperStates):
            elements = np.where(predicted_superstates[:,i] == ii)
            histogram_after_update[ii,i] = len(elements[0])
            
        weightscoeff[:,i+1] = 1 / maxNodes
        
    innovations       = predicted_state[:,i,:] - updated_state[:,i,:]
    min_innovation[i] = np.min(np.mean(np.abs(innovations[1:int(GSVDimension/2),:])))
    minCLB            = np.argmin(CLB[:,i])
    minCLBs[i]        = CLB[minCLB, i]
    
    
    #######################################################################
    ## PLOTTING
    
    if i % config['timeStepPlotting'] == 0: # Plotting every tot seconds
        

        ax1.plot(KLDabn_all[2:i])
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('KLDA')
        
        ax2.plot(min_innovation[2:i])
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('inn')
        
        ax3.plot(minCLBs[2:i])
        ax3.set_xlabel('time (s)')
        ax3.set_ylabel('CLB')
        
        max_inWindow = np.max((0, i-50)) # take last 50 time instants

        ax4.clear()
        ax4.scatter(data[max_inWindow:i, 0], data[max_inWindow:i, 1])
        ax4.set_xlim([min_pos_x,max_pos_x])
        ax4.set_ylim([min_pos_y,max_pos_y])
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        
        plt.show()
        plt.pause(0.0001)
        
        
        
        
        
        
        
        
        
    




