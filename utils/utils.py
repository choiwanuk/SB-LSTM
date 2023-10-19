from os import makedirs
from os.path import join, exists
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch


class bcolors :
    # Set Test Color
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fixSeed(seed) :
    # Fix Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def saveNetwork(opt, model, latest=False, best=False) :
    # Get Save Directory
    path = join(opt.checkpointsDir, opt.name, "models")
    mkdirs(path)
    if latest :
        torch.save(model.module.net.state_dict(), f"{path}/latest.pth")
    elif best :
        torch.save(model.module.net.state_dict(), f"{path}/best.pth")

def loadNetwork(network, saveType, opt) :
    # Get Path Directory
    saveFileName = f"{saveType}.pth"
    savePath = join(opt.checkpointsDir, opt.name, "models", saveFileName)
    
    # Load Network
    weights = torch.load(savePath)
    network.load_state_dict(weights)
    
    return network

def mkdirs(path) :
    # Make Directory
    if not exists(path) :
        makedirs(path)

# This function is inspired by [https://github.com/mdribeiro/DeepCFD]
def visualizeResult(iteration, target, output, error, saveDir, numSample=None, dpi=100) :
    # Create Directory
    mkdirs(saveDir)
    
    # Visualize Result
    target = np.transpose(target.detach().cpu().numpy(), (0, 1, 3, 2))
    output = np.transpose(output.detach().cpu().numpy(), (0, 1, 3, 2))
    
    for keys in error.keys() :
        if keys == "CFD" :
            error[keys] = np.transpose(error[keys].detach().cpu().numpy(), (0, 1, 3, 2))
    
    if numSample is not None :
        numSample = numSample
    else :
        numSample = len(target)
    
    matplotlib.use("agg")
    
    for i in range(numSample) :
        minu = np.min(target[i, 0, :, :])
        maxu = np.max(target[i, 0, :, :])
        
        minv = np.min(target[i, 1, :, :])
        maxv = np.max(target[i, 1, :, :])
        
        minp = np.min(target[i, 2, :, :])
        maxp = np.max(target[i, 2, :, :])
        
        mineu = np.min(error["CFD"][i, 0, :, :])
        maxeu = np.max(error["CFD"][i, 0, :, :])
        
        minev = np.min(error["CFD"][i, 1, :, :])
        maxev = np.max(error["CFD"][i, 1, :, :])
        
        minep = np.min(error["CFD"][i, 2, :, :])
        maxep = np.max(error["CFD"][i, 2, :, :])
        
        # Plot Result
        plt.figure(dpi = dpi)
        figure = plt.gcf()
        figure.set_size_inches(15, 10)
        plt.subplot(3, 3, 1)
        plt.title("CFD", fontsize=18)
        plt.imshow(np.transpose(target[i, 0, :, :]), cmap="jet", vmin = minu, vmax = maxu, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        plt.ylabel("Ux", fontsize=18)
        plt.subplot(3, 3, 2)
        plt.title("CNN", fontsize=18)
        plt.imshow(np.transpose(output[i, 0, :, :]), cmap="jet", vmin = minu, vmax = maxu, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        plt.subplot(3, 3, 3)
        plt.title("Error", fontsize=18)
        plt.imshow(np.transpose(error["CFD"][i, 0, :, :]), cmap="jet", vmin = mineu, vmax = maxeu, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")

        plt.subplot(3, 3, 4)
        plt.imshow(np.transpose(target[i, 1, :, :]), cmap="jet", vmin = minv, vmax = maxv, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        plt.ylabel("Uy", fontsize=18)
        plt.subplot(3, 3, 5)
        plt.imshow(np.transpose(output[i, 1, :, :]), cmap="jet", vmin = minv, vmax = maxv, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        plt.subplot(3, 3, 6)
        plt.imshow(np.transpose(error["CFD"][i, 1, :, :]), cmap="jet", vmin = minev, vmax = maxev, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")

        plt.subplot(3, 3, 7)
        plt.imshow(np.transpose(target[i, 2, :, :]), cmap="jet", vmin = minp, vmax = maxp, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        plt.ylabel("kinematic p", fontsize=18)
        plt.subplot(3, 3, 8)
        plt.imshow(np.transpose(output[i, 2, :, :]), cmap="jet", vmin = minp, vmax = maxp, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        plt.subplot(3, 3, 9)
        plt.imshow(np.transpose(error["CFD"][i, 2, :, :]), cmap="jet", vmin = minep, vmax = maxep, origin="lower", extent=[0,191,0,191])
        plt.colorbar(orientation="horizontal")
        
        # Save Figure
        plt.tight_layout()
        plt.savefig(f"{saveDir}/sample-{iteration}-{i}.png", dpi = dpi)
        
        # Clear Memory
        plt.close("all")
        
