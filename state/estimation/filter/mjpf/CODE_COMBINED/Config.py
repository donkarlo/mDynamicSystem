# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 17:20:59 2020

@author: asus
"""

def ConfigureSettings():
    
    config = {  # SETTINGS
              "nParticles"       : 10,
              "skewValue"        : 3,
              "plotBool"         : True,
              "printCounter"     : True,
              "testing"          : 1,             # 0=train, 1= AM, 2= UM
              "timeStepPlotting" : 5,             # after how many time steps to plot
                # FILES and FOLDERS
              "outputFolder"       : 'OUTPUT',
              "inputFolder"        : 'DATA',
              #"GSTrainFile"        : 'CL_data_train_odometry.mat',
              #"meanNodesFile"      : 'CL_clusterMeans_odometry.mat',
              #"DataColorNodeFile"  : 'CL_dataColorNode_odometry.mat'
              "Vocabulary"         : 'VocabularyMJPF.mat'
              }
    
    return config