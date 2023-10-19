from os import listdir
from os.path import join

import numpy as np

import torch
from torch.utils.data import Dataset


class CFDDataset(Dataset) :
    def __init__(self, opt, forMetrics) :
        # Inheritance
        super(CFDDataset, self).__init__()

        # Initialize Variables
        self.opt = opt
        self.forMetrics = forMetrics
        self.fileNameList, self.filePathList = self.getPathList()
    
    def __getitem__(self, index) :
        # Get Dataset Path
        dataPath = self.filePathList[index]
        
        # Load Dataset
        data = np.load(dataPath)
        
        # Transform Data
        input, target, mean = self.transforms(data)
        
        return {"input" : input, "target" : target, "mean" : mean, "name" : self.fileNameList[index]}
    
    def __len__(self) :
        return len(self.filePathList)

    def getPathList(self) :
        # Set Mode
        if self.opt.phase == "train" and not self.forMetrics :
            mode = "train"
        elif self.opt.phase == "train" and self.forMetrics :
            mode = "valid"
        elif self.opt.phase == "test" and self.forMetrics :
            mode = "test"
        else :
            raise ValueError("corresponding mode is not supported")
        
        # Get Absolute Parent Path of Dataset
        datasetPath = join(self.opt.dataRoot, mode)
        
        # Create List Instance for Adding File Name
        fileNameList, filePathList = [], []
        
        # Add File Name and Path
        for fileName in listdir(datasetPath) :
            fileNameList.append(fileName)
            filePathList.append(join(datasetPath, fileName))
            
        # Sort List
        fileNameList.sort()
        filePathList.sort()
        
        return fileNameList, filePathList
    
    def transforms(self, data) :
        # Slice Input Data
        input = data[0, :, :]
        # Slice Target Data
        target = data[1:4, :, :]
        # Convert Input and Target Data to PyTorch Tensor
        input = torch.FloatTensor(input).unsqueeze(0)
        target = torch.FloatTensor(target)
        # Convert Target Statistic Data to PyTorch Tensor
        mean = torch.FloatTensor([1, 1, 1]).unsqueeze(-1).unsqueeze(-1)
        
        return input, target, mean