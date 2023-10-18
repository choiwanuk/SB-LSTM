# Internal Flow Prediction in Arbitrary Shaped Channel Using Stream-wise Bidirectional LSTM
### [Paper] | [BibTex]

## Abstract

Deep learning (DL) methods have become the trend in predicting feasible solutions in a shorter time compared with traditional computational fluid dynamics (CFD) approaches. Recent researches have stacked numerous convolutional layers to extract high-level feature maps which are then used for the analysis of various shapes under differing conditions. However, these applications only deal with predicting the flow around the objects located near the center of the domain, whereas most fluid transport-related phenomena are associated with internal flow, such as pipe flow or air flows inside transportation vehicle engines. Hence, to broaden the scope of the DL approach in CFD, we introduced a stream-wise bidirectional (SB)-LSTM module that generates a better latent space from the internal fluid region by additionally extracting lateral connection features. To evaluate the effectiveness of the proposed method, we compared the results of using the SB-LSTM to the encoder-decoder(ED) model and the U-Net model with the results when not using it. When SB-LSTM was applied, in the qualitative comparison, it effectively addressed the issue of erratic fluctuations in the predicted field values. Furthermore, in terms of quantitative evaluation, the mean relative error (MRE) for the x-component of velocity, y-component of velocity, and pressure were reduced by at least 2.7%, 4.7%, and 15% respectively, compared to the absence of the SB-LSTM module. Furthermore, through a comparison of calculation time, it shows that our approach does not undermine the superiority of neural network's computational acceleration effect.

## Prerequisites
- Python 3.11.3
- PyTorch>=2.0.0
- Torchvision>=0.15.1
- NVIDIA GPU + CUDA cuDNN

- - ### Install PyTorch and dependencies from http://pytorch.org
- ### Please install dependencies by

- ## Dataset

## Training

## Test
