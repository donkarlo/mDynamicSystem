
import numpy as np
                         
# X : state
# P : covariance of state
# y : mesurement
# H : obs matrix
# R : observation covariance
def kf_update(X, P, y, H, R):
    
    # Find innovation and innovation covariance
    innovationMean = y - np.dot(H,X)
    H_transpose    = np.transpose(H)
    innovationCov  = np.dot(H, np.dot(P,H_transpose)) + R
    
    inverse_innovationCov = np.linalg.inv(innovationCov)
    
    # Kalman Gain
    kalmanGain = np.dot(P, np.dot(H_transpose, inverse_innovationCov))
    
    X = X + np.dot(kalmanGain,innovationMean)
    
    # We also update the covariance estimation:
    covEstimatedTemp = np.dot(kalmanGain, innovationCov)
    #covEstimatedTemp = np.dot(kalmanGain, predCovariance)
    covEstimatedTemp = np.dot(covEstimatedTemp, np.transpose(kalmanGain))
    P = P - covEstimatedTemp    
    
    return X, P
    

# X : state
# P : covariance of state
# A : transition matrix
# Q : prediction covariance matrix
# B : control matrix
# u : control vector (velocity)

def kf_predict(X,P,A,Q,B,u):
    '''
    X = currentState
    P = currentCov
    A = A
    Q = Q2
    B = B
    u = U
    '''
    
    X = np.dot(A, X) + np.dot(B,u)
    P = np.dot(A, np.dot(P, np.transpose(A))) + Q
       
    return X, P