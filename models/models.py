import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import utils
from models.networks import networks
from models.loss import *


class SBLSTM_network(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(SBLSTM_network, self).__init__()
        
        # Initialize Variables
        self.OKBLUE, self.ENDC = utils.bcolors.OKBLUE, utils.bcolors.ENDC
        
        # Create Model Instance
        self.opt = opt
        self.net = networks(opt)
        
        # Fix Seed
        utils.fixSeed(self.opt.seed)
        
        self.get_n_params()
        # Initialize Model
        self.initializeNetwork()
        # Check Whether to Load Model
        self.loadCheckpoints()

    def forward(self, input, target, mean, mode) :
        if mode == "train" :
            self.net.train()
            loss, output, target = self.computeLoss(input, target, mean, mode)
            return loss, output, target
        elif mode == "inference" :
            self.net.eval()
            with torch.no_grad() :
                loss, output, target = self.computeLoss(input, target, mean, mode)
            return loss, output, target
        else :
            raise ValueError(f"{mode} is not supported")

    def loadCheckpoints(self) :
        if self.opt.phase == "test" :
            saveType = self.opt.saveType
            self.net = utils.loadNetwork(self.net, saveType, self.opt)

        
    ############################################################################
    # Private helper methods
    ############################################################################

    def get_n_params(self):
        model = self.net 
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        print("Total params : " + str(pp))
       

    def initializeNetwork(self) :
        def init_weights(m, initType="normal", gain=0.02) :
            className = m.__class__.__name__
            # Initialize Convolution Weights
            if hasattr(m, "weight") and className.find("Conv") != -1 :
                if initType == "normal" :
                    init.normal_(m.weight.data, 0.0, gain)
                elif initType == "xavier" :
                    init.xavier_normal_(m.weight.data, gain = gain)
                elif initType == "xavier_uniform" :
                    init.xavier_uniform_(m.weight.data, gain = gain)
                elif initType == 'kaiming' :
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif initType == 'orthogonal' :
                    init.orthogonal_(m.weight.data, gain=gain)
                elif initType == 'none' :
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f"{initType} method is not supported")
                if hasattr(m, "bias") and m.bias is not None :
                    init.constant_(m.bias.data, 0.0)

        # Create List Instance for Adding Network
        if self.opt.phase == "train" :
            networkList = [self.net]
        else :
            networkList = [self.net]
        
        #I Initialize Network Weights
        for network in networkList :
            network.apply(init_weights)

    def computeLoss(self, input, target, mean, mode) :
        # Create Dictionary Instance for Adding Loss
        loss = {}
        
        # Get Inference Result
        output = self.getCFDOutput(input)

        # Compute Training Loss
        if mode == "train" :
            loss["CFD"] = PixelLoss(output, target, mean, self.opt.useAffine)
        
        elif mode == "inference" :
            loss["CFD"] = PixelLoss(output, target, mean, self.opt.useAffine)
            #loss["direction"] = directional_loss(output[:,:2], target[:,:2], mean, input)

        output = output*mean
        target = target*mean
        
        return loss, output, target
            
    def getCFDOutput(self, input):
        # Get Inference Result
        output = self.net(input)
        
        return output
    
    def useGPU(self) :
        return self.opt.gpuIds != "-1"


def assignOnMultiGpus(opt, model):
    # If Use CUDA
    if opt.gpuIds != "-1" :
        gpus = list(map(int, opt.gpuIds.split(",")))
        model = model.cuda(gpus[0])
        model = nn.DataParallel(model, device_ids=gpus)
    else:
        model.module = model
    assert len(opt.gpuIds.split(",")) == 0 or opt.batchSize % len(opt.gpuIds.split(",")) == 0
    
    return model


def preprocessInput(opt, input, target, mean) :
    # If Use CUDA
    if opt.gpuIds != "-1" :
        gpus = list(map(int, opt.gpuIds.split(",")))
        input = input.cuda(gpus[0])
        target = target.cuda(gpus[0])
        mean = mean.cuda(gpus[0])
    
    # Standardize Target Tensor
    target = target/mean

    return input, target, mean