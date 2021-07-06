# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 01:52:33 2019

@author: Dell
"""

"""
File containing scripts to download audio from various datasets

Also has tools to convert audio into numpy
"""
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

import audioUtils


###################
# Google Speech Commands Dataset V2
###################


GSCmdV2Categs = {'Noise' : 0,  'Left' : 1, 'Right' : 2}
numGSCmdV2Categs = 3

def PrepareGoogleSpeechCmd(version = 2, forceDownload = False, task = 'leftright'):
    """
    Prepares Google Speech commands dataset version 2 for use
    
    tasks: leftright 
    
    Returns full path to training, validation and test file list and file categories
    """
    allowedTasks = [ 'leftright']
    if task not in allowedTasks:
        raise Exception('Task must be one of: {}'.format(allowedTasks))
    
    basePath = 'D:/Thesis  work/speech recog with lstm attention/speechRecogination/sd_GSCmdV22'
  
        
    if task=='leftright':
        GSCmdV2Categs = {'Noise' : 0, 'Left' : 1, 'Right' : 2}
        numGSCmdV2Categs = 3
        
     
    print('Converting test set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/test/')
    print('Converting training set WAVs to numpy files')
    audioUtils.WAV2Numpy(basePath + '/train/')
  #  Az=basePath+'/train/'
    #read split from files and all files in folders
    testWAVs = pd.read_csv(basePath+'/train/testing_nlr111.txt', sep=" ", header=None)[0].tolist()
    #if (path.exist())
    valWAVs  = pd.read_csv(basePath+'/train/validation_nlr111.txt', sep=" ", header=None)[0].tolist()

    testWAVs = [os.path.join(basePath+'/train/', f ) for f in testWAVs if f.endswith('.npy')]
    valWAVs  = [os.path.join(basePath+'/train/', f ) for f in valWAVs if f.endswith('.npy')]
    allWAVs  = []
    for root, dirs, files in os.walk(basePath+'/train/'):
        allWAVs += [root+'/'+ f for f in files if f.endswith('.wav.npy')]
    trainWAVs = list( set(allWAVs)-set(valWAVs)-set(testWAVs) )

    testWAVsREAL = []
    for root, dirs, files in os.walk(basePath+'/test/'):
        testWAVsREAL += [root+'/'+ f for f in files if f.endswith('.wav.npy')]

    #get categories
    testWAVlabels     = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVs]
    valWAVlabels      = [_getFileCategory(f, GSCmdV2Categs) for f in valWAVs]
    trainWAVlabels    = [_getFileCategory(f, GSCmdV2Categs) for f in trainWAVs]
    testWAVREALlabels = [_getFileCategory(f, GSCmdV2Categs) for f in testWAVsREAL]
    

    
    #build dictionaries
    testWAVlabelsDict     = dict(zip(testWAVs, testWAVlabels))
    valWAVlabelsDict      = dict(zip(valWAVs, valWAVlabels))
    trainWAVlabelsDict    = dict(zip(trainWAVs, trainWAVlabels))
    testWAVREALlabelsDict = dict(zip(testWAVsREAL, testWAVREALlabels))
    
    #a tweak here: we will heavily underuse silence samples because there are few files.
    #we can add them to the training list to reuse them multiple times
    #note that since we already added the files to the label dicts we don't need to do it again
    
    #for i in range(200):
    #    trainWAVs = trainWAVs + backNoiseFiles
    
    #info dictionary
    trainInfo = {'files' : trainWAVs, 'labels' : trainWAVlabelsDict}
    valInfo = {'files' : valWAVs, 'labels' : valWAVlabelsDict}
    testInfo = {'files' : testWAVs, 'labels' : testWAVlabelsDict}
    testREALInfo = {'files' : testWAVsREAL, 'labels' : testWAVREALlabelsDict}
    gscInfo = {'train' : trainInfo, 'test' : testInfo, 'val' : valInfo, 'testREAL' : testREALInfo}    
    
    print('Done preparing Google Speech commands dataset version {}'.format(version))
    
    return gscInfo, numGSCmdV2Categs
    
    
def _getFileCategory(file, catDict):
    """
    Receives a file with name sd_GSCmdV2/train/<cat>/<filename> and returns an integer that is catDict[cat]
    """
    categ = os.path.basename(os.path.dirname(file))
    return catDict.get(categ,0)

    