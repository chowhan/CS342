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
		self.conv1 = nn.Conv2d(3, 9, 5, 2, 1)
		self.bn1 = nn.BatchNorm2d(9)
		self.conv2 = nn.Conv2d(9, 27, 5, 2, 1)
		self.bn2 = nn.BatchNorm2d(27)
		self.conv3 = nn.Conv2d(27, 81, 5, 2, 1)
		self.bn3 = nn.BatchNorm2d(81)
		self.conv4 = nn.Conv2d(81, 243, 4, 2, 1)
		self.bn4 = nn.BatchNorm2d(243)

		self.upconv1 = nn.ConvTranspose2d(243, 81, 5, 2, 1)
		self.upconv2 = nn.ConvTranspose2d(81, 27, 5, 2, 1)
		self.upconv3 = nn.ConvTranspose2d(27, 9, 5, 2, 1)
		self.upconv4 = nn.ConvTranspose2d(9, 6, 4, 2)

		self.relu = nn.ReLU(True)

	def forward(self, x):
		
		'''
		Your code here
		'''
		c1 = self.conv1(x)
		c1 = self.bn1(c1)
		c1 = self.relu(c1)

		c2 = self.conv2(c1)
		c2 = self.bn2(c2)
		c2 = self.relu(c2)

		c3 = self.conv3(c2)
		c3 = self.bn3(c3)
		c3 = self.relu(c3)

		c4 = self.conv4(c3)
		c4 = self.bn4(c4)
		c4 = self.relu(c4)

		u1 = self.upconv1(c4)
		u2 = self.upconv2(u1 + c3)
		u3 = self.upconv3(u2 + c2)
		u4 = self.upconv4(u3 + c1)

		return u4
