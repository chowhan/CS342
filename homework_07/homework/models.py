from torch import nn

class FConvNetModel(nn.Module):

	"""
	Define your fully convolutional network here
	"""

	def __init__(self):
		
		super().__init__()
		
		'''
		Your code here
		'''
		self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
		self.conv2 = nn.Conv2d(32, 16, 3, 2)
		self.upconv1 = nn.ConvTranspose2d(16, 8, 3, 2, 2)
		self.upconv2 = nn.ConvTranspose2d(8, 6, 3, 2, 1)

	def forward(self, x):
		
		'''
		Your code here
		'''
		c1 = self.conv1(x)
		c2 = self.conv2(c1)
		up1 = self.upconv1(c2)
		up2 = self.upconv2(up1 + c1)
		return up2
