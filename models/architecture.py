import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Encoder, self).__init__()
        
        # Initialize Variable
        channels = opt.channels
        
        # Create Encoder Layer Instance
        self.convIn = nn.Conv2d(opt.inputDim, channels, kernel_size=3, stride=1, padding=1)
        self.down0 = Downscale2D(channels, channels*2)
        self.down1 = Downscale2D(channels*2, channels*4)
        self.down2 = Downscale2D(channels*4, channels*8)
        self.down3 = Downscale2D(channels*8, channels*16)
        self.lstm = Streamwise_BRNN(channels*16,32,24) #input cnn channel_size, LSTM state size, width_size
        
    def forward(self, input) :
        e0 = self.convIn(input)
        e1 = self.down0(e0)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.lstm(e4)
        return e0, e1, e2, e3, e4, e5
    
class Streamwise_BRNN(nn.Module):

    def __init__(self,inchannel_size, hidden_size, channel_size):
        super(Streamwise_BRNN, self).__init__()

        self.inpointconv = Pointwise2D(inchannel_size,hidden_size)
        self.biLSTM = BRNN(int(hidden_size*channel_size),  int(hidden_size*channel_size/2), 1)
        self.outpointconv = Pointwise2D(int(hidden_size*channel_size),int(inchannel_size/2))

    def forward(self, x):
        batch_size, channel_size, height_size, width_size =x.shape
        
        #in-pointwiseconv
        output = self.inpointconv(x)
        
        #flattening
        output = output.permute(0,3,1,2)
        output = output.reshape(batch_size,width_size,-1)

        #SB-LSTM-module
        output = self.biLSTM(output)

        #repeat H-times
        output = output.unsqueeze(-1)        
        output = output.expand(-1,-1,-1,12)
        out = output.permute(0,2,3,1)
        
        #out-pointwiseconv
        out = self.outpointconv(out)
        
        return out

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        
        
        
    def forward(self, x):
        #initialize LSTM
        self.h0 = torch.zeros(self.num_layers*2 , x.size(0), self.hidden_size).cuda()
        self.c0 = torch.zeros(self.num_layers*2 , x.size(0), self.hidden_size).cuda()

        out, (hidden,cell) = self.lstm(x,(self.h0,self.c0))
        
        return out
    
    
class Decoder(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(Decoder, self).__init__()
        
        # Initialize Variable
        channels = opt.channels
        
        # Create Decoder Layer Instance
        self.up0 = Upscale2D(channels*16+channels*8+channels*8, channels*8)
        self.up1 = Upscale2D(channels*8+channels*4, channels*4)
        self.up2 = Upscale2D(channels*4+channels*2, channels*2)
        self.up3 = Upscale2D(channels*2+channels, channels)
        self.convOut = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
            
    def forward(self, skipConnection) :
        e0, e1, e2, e3, e4, e5 = skipConnection
        d0 = self.up0(torch.cat([e4,e5], dim=1), e3)
        d1 = self.up1( d0, e2)
        d2 = self.up2(d1, e1)
        d3 = self.up3(d2, e0)
        d4 = self.convOut(d3)
        return d4
        
    
class Downscale2D(nn.Module) :
    def __init__(self, inChannels, outChannels) :
        # Inheritance
        super(Downscale2D, self).__init__()
        
        # Create Convolution Layer Instance
        self.down = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=2, padding=1) 
        self.conv0 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)

    def forward(self, input) :
        output = self.down(input)
        output = self.conv0(F.leaky_relu(output, 0.2))
        output = self.conv1(F.leaky_relu(output, 0.2))
        
        return output

class Upscale2D(nn.Module) :
    def __init__(self, inChannels, outChannels) :
        # Inheritance
        super(Upscale2D, self).__init__()
        
        # Create Convolution Layer Instance        
        self.conv0 = nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1)
        
    def forward(self, input, skipConnection) :
        output = F.interpolate(input, scale_factor=2)
        output = F.leaky_relu(self.conv0(torch.cat([output, skipConnection], dim=1)), 0.2)
        output = F.leaky_relu(self.conv1(output), 0.2)
        
        return output

class Pointwise2D(nn.Module) :
    def __init__(self, inChannels, outChannels) :
        # Inheritance
        super(Pointwise2D, self).__init__()
        
        # Create Convolution Layer Instance
        self.down = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0) 

    def forward(self, input) :
        output = self.down(input)
        output = F.leaky_relu(output, 0.2)
        
        return output
