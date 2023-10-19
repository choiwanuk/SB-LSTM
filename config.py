from utils import utils
import argparse



def readArguments(train=True) :
    # Set All Arguments
    parser = argparse.ArgumentParser()
    parser = addAllArguments(parser, train)
    parser.add_argument("--phase", type=str, default="train")
    opt = parser.parse_args()
    opt.phase = "train" if train else "test"
    
    utils.fixSeed(opt.seed)
    
    return opt


def addAllArguments(parser, train) :
    # General Options
    parser.add_argument("--name", type=str, default="UNet-with-SBLSTM", help="name of the experiment. It decides where to store samples and models")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpuIds", type=str, default="0,1", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
    parser.add_argument("--checkpointsDir", type=str, default="./checkpoints", help="models are saved here")
    parser.add_argument("--batchSize", type=int, default=64, help="input batch size")
    parser.add_argument("--dataRoot", type=str, default="./dataset/", help="path to dataset root")
    parser.add_argument("--datasetMode", type=str, default="CFDDataset", help="this option indicates which dataset should be loaded")
    parser.add_argument("--numWorkers", type=int, default=10, help="num_workers argument for dataloader")

    parser.add_argument("--channels", type=int, default=32, help="# of autoencoder filters in first conv layer in generator")
    parser.add_argument("--inputDim", type=int, default=1, help="number of dimension of input features")
    parser.add_argument("--initType", type=str, default="xavier", help="selects weight initialization type")

    if train :
        # For Training
        parser.add_argument("--numEpochs", type=int, default=200, help="number of epochs to train")
        parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
        parser.add_argument("--beta2", type=float, default=0.999, help="momentum term of adam")
        parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
        parser.add_argument("--decayRate", type=float, default=1e-2, help="rate for cosine annealing")
        parser.add_argument("--useAffine", action="store_true", help="if specified, use affine in pixel loss")
        parser.add_argument("--resultsDir", type=str, default="./results/", help="saves testing results here.")
        parser.add_argument("--saveType", type=str, default="best", help="which epoch to load to evaluate a model")
        

    else :
        # For Testing
        parser.add_argument("--resultsDir", type=str, default="./results/", help="saves testing results here.")
        parser.add_argument("--saveType", type=str, default="best", help="which epoch to load to evaluate a model")
        parser.add_argument("--useAffine", action="store_true", help="if specified, use affine in pixel loss")
    return parser