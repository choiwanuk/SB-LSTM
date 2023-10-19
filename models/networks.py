import torch
from torch import nn

from models.architecture import Encoder, Decoder



class networks(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(networks, self).__init__()
        
        # Initialize Variable
        self.opt = opt
        
        # Create networks' Layer Instance
        self.encoder = Encoder(opt)
        self.uxDecoder = Decoder(opt)
        self.uyDecoder = Decoder(opt)
        self.pDecoder = Decoder(opt)
    
    def forward(self, input) :
        output = self.encoder(input)
       
        outputUx = self.uxDecoder(output)

        # Concatenate All Outputs
        output = torch.cat([outputUx, 
                            self.uyDecoder(output), 
                            self.pDecoder(output)], dim=1)
        
        return output
