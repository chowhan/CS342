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
    
    def forward(self, x):
        '''
        Your code here
        '''
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
        self.conv1 = nn.Conv2d(3, 16, 5)
        # You can wrap all the layers using the nn.Sequential which will make your

        self.res_block1 = nn.Sequential(
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(16),
        )

        self.conv2 = nn.Conv2d(16, 32, 5)

        self.res_block2 = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(32),
        )

        self.conv3 = nn.Conv2d(32, 64, 5)

        self.res_block3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(64),
        )

        self.conv4 = nn.Conv2d(64, 128, 5)

        self.res_block4 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(True),
            torch.nn.BatchNorm2d(128),
        )

        self.conv5 = nn.Conv2d(128, 12, 5)
        self.pool = nn.AvgPool2d(2)
        self.linear = nn.Linear(5508, 64)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        Your code here
        '''
        
        x = self.conv1(x)
        h1 = self.res_block1(x)
        x = self.conv2(h1 + x)
        h2 = self.res_block2(x)
        x = self.conv3(h2 + x)
        h3 = self.res_block3(x)
        x = self.conv4(h3 + x)
        h4 = self.res_block4(x)
        x = self.conv5(h4 + x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
