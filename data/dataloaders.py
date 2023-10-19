import importlib

from torch.utils.data import DataLoader


def selectDatasetName(mode) :
    # Select Dataset Name
    if mode == "CFDDataset" :
        return "CFDDataset"
    else :
        raise ValueError(f"No such dataset as {mode}")


def getDataloaders(opt) :
    # Select Dataset Name
    datasetName = selectDatasetName(opt.datasetMode)
    
    # Import Python Code
    fileName = importlib.import_module(f"data.{datasetName}")
    
    # Create Dataset Instance
    if opt.phase == "train":
        trainDataset = fileName.__dict__[datasetName](opt, forMetrics=False)
        validDataset = fileName.__dict__[datasetName](opt, forMetrics=True)
        
        # Train and Valid PyTorch DataLoader Instance
        trainDataLoader = DataLoader(trainDataset, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=opt.numWorkers)
        validDataLoader = DataLoader(validDataset, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=opt.numWorkers)
        
        return trainDataLoader, validDataLoader
    
    elif opt.phase == "test" :
        testDataset = fileName.__dict__[datasetName](opt, forMetrics=True)
        
        # Test PyTorch DataLoader Instance
        testDataLoader = DataLoader(testDataset, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=opt.numWorkers)
        
        return testDataLoader
    
    else :
        raise ValueError(f"No such phase as {opt.phase}")