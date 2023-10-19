import config
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch


from data import dataloaders
from utils import utils
from models import models


def main() :
    # Read Options
    opt = config.readArguments(train=False)
    
    # Create DataLoader Instance
    testDataLoader = dataloaders.getDataloaders(opt)
    
    # Create Model Instance
    model = models.SBLSTM_network(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    # Create Save Root Path
    savePath = f"{opt.resultsDir}/{opt.name}"
    
    # Create Path for Each Folder
    npySavePath = f"{savePath}/npy"
    
    # Create Directory
    utils.mkdirs(npySavePath)
    
    
    # Create Dictionary Instance for Adding Results
    data = {"Name":[], "Ux MAE":[], "Uy MAE":[], "p MAE":[], "Average MAE":[]}
    flowRate = {}
    
    # Create Test tqdm Bar
    testBar = tqdm(testDataLoader)
    
    for testData in testBar :
        # Load Dataset
        input, target, mean, fileName = testData["input"], testData["target"], testData["mean"], testData["name"]
        input, target, mean = models.preprocessInput(opt, input, target, mean)
        
        loss, output, target = model(input, target, mean, mode="inference")
       
        # Add Result to Dictionary
        
        for i in range(len(target)) :
            combinedOutput = torch.cat([input[i],target[i,:,:,:],output[i,:,:,:]], dim=0).cpu().numpy()
            np.save(f"{npySavePath}/{fileName[i].replace('pkl', 'npy')}", combinedOutput)


if __name__ == "__main__" :
    main()