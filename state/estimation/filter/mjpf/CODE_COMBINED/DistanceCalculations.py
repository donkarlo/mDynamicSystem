
import numpy as np

def KLDiv(p, q):
    """Kullback-Liebler divergence from multinomial p to multinomial q,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Clip before taking logarithm to avoid NaNs (but still exclude
    # zero-probability mixtures from the calculation)
    return (p * (np.log(p.clip(1e-10,1))
                 - np.log(q.clip(1e-10,1)))).sum(axis)   # - N
    
def single_KLD_Abnormality(PP, QQ, KLDAbnMax, histogramProb, N):
    
    if np.sum(np.isinf(QQ)) >= 1:
        KLD_simmetrica = KLDAbnMax 
    elif np.sum(np.isnan(QQ)) >= 1:
        KLD_simmetrica = KLDAbnMax
    elif np.sum(np.isnan(PP)) >= 1:
        KLD_simmetrica = KLDAbnMax
    else:
        KLD_simmetrica = (histogramProb/N)*KLDiv(PP,QQ) + (histogramProb/N)*KLDiv(QQ,PP) #to achieve symmerty
    
    if np.sum(np.isinf(KLD_simmetrica)) >= 1:
        KLD_simmetrica = KLDAbnMax;
    
    return KLD_simmetrica

## Input:
# totNumOfSuperstates: total number of Superstates
# N: total number of Particles
# histogram: histogram at time t-1 (after PF resampling)
# transitionMat: the transition matrix learned from previous experience
# probability_lamdaS: probability vector representing a discrete probability disctribution         
def KLD_Abnormality(nSuperstates, N, histogram, transitionMat, probability_lamdaS, KLDAbnMax):
    
    ##Procedure:
    sommaKLD_simmetrica = 0
    
    for indKLD in range(nSuperstates):
        particella = histogram[indKLD]
        
        if particella > 0:
            
            PP = transitionMat[indKLD,:] +1e-20 # add 1e-100 since KLD doesnt allow zero values
            QQ = probability_lamdaS
            
            KLD_simmetrica = single_KLD_Abnormality(PP, QQ, KLDAbnMax, particella, N)
            
            sommaKLD_simmetrica = sommaKLD_simmetrica + KLD_simmetrica
        
    return sommaKLD_simmetrica



def CalculateBhattacharyyaDistance(pm, pv, qm, qv):
    
    # Copyright (c) 2008 Carnegie Mellon University
    #
    # You may copy and modify this freely under the same terms as
    # Sphinx-III
    
    #__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
    #__version__ = "$Revision$"

    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    """
    
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = np.log(pv).sum()
    ldqv = np.log(qv).sum(axis)
    # Log-determinant of pqv
    ldpqv = np.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    
    return dist + norm

def calculateLamdaS(nSuperStates, measurement, meanOfSuperstates, R, covarianceOfSuperstates, skewValue):
    
    lamdaS = np.zeros(nSuperStates)
    # Calculate lambda in terms of battacharyya distance b/w observation & each superstate:
    for index_s in range(nSuperStates):
        
        varianceOfMeasurement_s = np.diagonal(R)
        varianceOfSuperstate_s  = np.diagonal(covarianceOfSuperstates[index_s])
        
        lamdaS[index_s] = CalculateBhattacharyyaDistance(measurement,varianceOfMeasurement_s,meanOfSuperstates[index_s, :],varianceOfSuperstate_s)
              
    #Convert lamda to a discrete probability distribution:
    # using skewValue can help to make the probability distribution more skewed (for example if n=1 give you [0.6 0.4], n=2 will give you [0.1 0.9])
    probability_lamdaS = (1/(np.power(lamdaS, skewValue)))/np.sum(1/(np.power(lamdaS, skewValue)))
    
    return probability_lamdaS