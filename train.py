import wandb
from tqdm import tqdm
import torch
from torch import optim
import config
from data import dataloaders
from utils import utils
from models import models

def main() : 
    # Read Options
    opt = config.readArguments(train=True)
    
    
    # Create DataLoader Instance
    trainDataLoader, validDataLoader = dataloaders.getDataloaders(opt)

    # Create Model Instance
    model = models.SBLSTM_network(opt)
    model = models.assignOnMultiGpus(opt, model)
    
    
    # Create Optimizer Instance
    optimizer = optim.Adam(model.module.net.parameters(), 
                           lr=opt.lr, 
                           betas=(opt.beta1, opt.beta2))
    
    # Create Scheduler Instance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=opt.numEpochs, 
                                                     eta_min=opt.lr*opt.decayRate)
    
    # Create AverageMeter Instance for Updating Metrics
    pixelTrainLoss = utils.AverageMeter()
    streamTrainLoss = utils.AverageMeter()
    directionTrainLoss = utils.AverageMeter()
    uxValidLoss, uyValidLoss, u_direction_val_Loss, pValidLoss = utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter(), utils.AverageMeter()
    conValidLoss, nsValidLoss = utils.AverageMeter(), utils.AverageMeter()
    
    # Initialize Variable for Saving Best Model
    bestLoss = torch.inf
    
    # Train Model
    for epoch in range(opt.numEpochs) :
        # Create Training tqdm Bar
        trainBar = tqdm(trainDataLoader)
        
        # Reset AverageMeter Instance
        pixelTrainLoss.reset()
        streamTrainLoss.reset()
        directionTrainLoss.reset()
        for data in trainBar :
            # Load Dataset
            input, target, mean = data["input"], data["target"], data["mean"]
            input, target, mean = models.preprocessInput(opt, input, target, mean)
            
            # Update Model Parameters
            optimizer.zero_grad()
            loss, output, target = model(input, target, mean, mode="train")

            # Compute Loss
            pixelTrainLoss.update(loss["CFD"].mean().item(), opt.batchSize)
            
            # Back-Propagation
            loss = sum(loss.values()).mean()
            loss.backward()
            optimizer.step()
            
            # Update Training tqdm Bar
            trainBar.set_description(desc=f"[{epoch}/{opt.numEpochs-1}] [Train] < Pixel:{pixelTrainLoss.avg*1e5:.4f}×1E-5  >")
                
        # Create Validation tqdm Bar
        validBar = tqdm(validDataLoader)
        
        # Reset AverageMeter Instance
        uxValidLoss.reset()
        uyValidLoss.reset()
        pValidLoss.reset()
        
        
        for iteration, data in enumerate(validBar) :
            # Load Dataset
            input, target, mean = data["input"], data["target"], data["mean"]
            input, target, mean = models.preprocessInput(opt, input, target, mean)
            
            # Inference Result
            loss, output, target = model(input, target, mean, mode="inference")
            
        
            uxValidLoss.update(loss["CFD"][:, 0, :, :].mean().item(), len(loss))
            uyValidLoss.update(loss["CFD"][:, 1, :, :].mean().item(), len(loss))
            pValidLoss.update(loss["CFD"][:, 2, :, :].mean().item(), len(loss))
            
            # Compute Governing Equation Loss
            
            # Compute Average Loss
            avgValidLoss = (uxValidLoss.avg + uyValidLoss.avg + pValidLoss.avg)/3
            utils.visualizeResult(iteration, target, output, loss, 
                                  f"{opt.checkpointsDir}/{opt.name}/inference/{epoch}/", 2)
            
            
            # Update Validation tqdm Bar
            validBar.set_description(desc=f"[{epoch}/{opt.numEpochs-1}] [Valid] < Avg.:{avgValidLoss*1e5:.4f}×1E-5 | Ux:{uxValidLoss.avg*1e5:.4f}×1E-5 | Uy:{uyValidLoss.avg*1e5:.4f}×1E-5 | p:{pValidLoss.avg*1e5:.4f}×1E-5 >")
        
        # Save Model
        if avgValidLoss < bestLoss :
            bestLoss = avgValidLoss
            utils.saveNetwork(opt, model, best=True)
        utils.saveNetwork(opt, model, latest=True)

        # Update Learning Rate
        scheduler.step()


if __name__ == "__main__" :
    main()