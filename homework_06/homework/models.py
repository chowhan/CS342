import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class Block(nn.Module):
    '''
    Your code for resnet blocks
    '''
    def __init__(self, in_channel, bottle_channel, out_channel, stride):
        super(Block, self).__init__()
        '''
        Your code here
        '''
        # self.model = nn.Sequential(
        #     nn.Conv2d(3, 32, 5, 2, 1),
        #     nn.ReLU(True),
        #     torch.nn.GroupNorm(4,32, affine=False),
        #     nn.Conv2d(32, 64, 5, 2, 1),
        #     nn.ReLU(True),
        #     torch.nn.GroupNorm(4,64, affine=False),
        #     nn.Conv2d(64, 128, 5, 2, 1),
        #     nn.ReLU(True),
        #     torch.nn.GroupNorm(4,128, affine=False),
        #     nn.Conv2d(128, 6, 5, 2, 1),
        #     nn.AvgPool2d(3)
        # )

        self.conv1 = nn.Conv2d(in_channel, out_channel, 5, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, (out_channel + 16), 5, stride, 1)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3 = nn.Conv2d((out_channel + 16), in_channel, 5, stride, 1)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.stride = stride
    
    def forward(self, x):
        '''
        Your code here
        '''
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(z)

        x += res
        x = self.relu(x)
        return x
        
    

class ConvNetModel(nn.Module):
    '''
    Your code for the model that computes classification from the inputs to the scalar value of the label.
    Classification Problem (1) in the assignment
    '''
    
    def __init__(self):
        super(ConvNetModel, self).__init__()
        '''
        Your code here
        '''
        self.layer1 = Block()
    
    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        Your code here
        '''
        
        return x
