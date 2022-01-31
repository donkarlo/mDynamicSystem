
# This script contains functions for calculating probabilistic distances

import numpy as np
import torch

import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)



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

'''
def CalculateBhattacharyyaDistanceTorch(mu1, C1, mu2, C2):
    #C=(C1+C2)/2;
    #dmu=(mu1-mu2)/chol(C);
    #% % dmu=(mu1-mu2)/lu(C); % akr: or qr ma chol molto meglio
    #try
    #
    #    d=0.125*dmu*dmu'+0.5*log(det(C/chol(C1*(diag(C2).*eye(size(C2, 1))))));
    #    
    #catch
    #    d=0.125*dmu*dmu'+0.5*log(abs(det(C/sqrtm(C1*C2))));
    #    %warning('MATLAB:divideByZero','Data are almost linear dependent. The results may not be accurate.');
    #end
    
    C   = (C1+C2)/2
    dmu = (mu1-mu2)/torch.cholesky(C)
    
    d   = 0.125*dmu*torch.transpose(dmu, 1,0)
    
    try:
        
        print('try')
    
        # (diag(C2).*eye(size(C2, 1)))
        dimState             = C2.shape[0]
        eyeMatrix            = torch.eye(dimState).to(device)
        diagonalOfC2         = torch.diag(C2)
        diagonalMatrixOfC2   = diagonalOfC2*eyeMatrix
        
        # (C1*(diag(C2).*eye(size(C2, 1))))
        matrixMultiplication = torch.matmul(C1.float(), diagonalMatrixOfC2.float())
        
        # chol(C1*(diag(C2).*eye(size(C2, 1))))
        choleskyOnMul        = torch.cholesky(matrixMultiplication)
        
        print('matrixMultiplication')
        print(matrixMultiplication)
        
        print('choleskyOnMul')
        print(choleskyOnMul)
        
        # C/chol(C1*(diag(C2).*eye(size(C2, 1)))))
        divisionCandChol     = torch.lstsq(C,choleskyOnMul)
        
        print('divisionCandChol')
        print(divisionCandChol)
    
        d = d + 0.5*torch.log(torch.det(divisionCandChol))
        
        print('d')
        print(d)
    
    except:
        
        # d=0.125*dmu*dmu'+0.5*log(abs(det(C/sqrtm(C1*C2))));
        
        print('exception')
        
        squareRootOfMatrix         = torch.sqrt(torch.matmul(C1.float(), C2.float()))
        divisionOnSquareRootMatrix = torch.lstsq(C.float(),squareRootOfMatrix.float())
        
        print(divisionOnSquareRootMatrix)
        
        d = d + 0.5*torch.log(torch.abs(torch.det(divisionOnSquareRootMatrix)))
        
    return
'''


def CalculateBhattacharyyaDistanceTorch(pm, pv, qm, qv):
    
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
    
    ldpv  = torch.log(pv).sum()
    ldqv  = torch.log(qv).sum(axis)
    # Log -determinant of pqv
    ldpqv = torch.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    
    return dist + norm    