from torch import nn
import torch.nn.functional as F
import torch

def one_hot(x, n=6):
	batch_size, h, w= x.size()
	x = (x.view(-1,h,w,1) == torch.arange(n, dtype=x.dtype, device=x.device)[None]).float() - torch.as_tensor([0.6609, 0.0045, 0.017, 0.0001, 0.0036, 0.314], dtype=torch.float, device=x.device)
	x = x.permute(0,3,1,2)
	return x
	

class FConvNetModel(nn.Module):

	"""
	Define your fully convolutional network here
	"""

	def __init__(self):
		
		super().__init__()
		
		'''
		Your code here
		'''
		self.conv1 = nn.Conv2d(4, 32, 5, 2, 2)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, 5, 2, 2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, 5, 2, 2)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(128, 256, 5, 2, 2)
		self.bn4 = nn.BatchNorm2d(256)
		# self.conv5 = nn.Conv2d(256, 512, 5, 2, 2)
		# self.bn5 = nn.BatchNorm2d(512)

		# self.upconv1 = nn.ConvTranspose2d(512, 256, 5, 2, 2, 1)
		self.upconv2 = nn.ConvTranspose2d(256, 128, 5, 2, 2, 1)
		self.upconv3 = nn.ConvTranspose2d(128, 64, 5, 2, 2, 1)
		self.upconv4 = nn.ConvTranspose2d(64, 32, 5, 2, 2, 1)
		self.upconv5 = nn.ConvTranspose2d(32, 3, 5, 2, 2, 1)

		nn.init.constant_(self.upconv4.weight, 0)
		nn.init.constant_(self.upconv4.bias, 0)

		self.relu = nn.LeakyReLU(inplace=True)


	def forward(self, image, labels):
		
		'''
		Your code here
		'''
		hr_image = nn.functional.interpolate(image, scale_factor=4, mode='nearest')
		labels = torch.unsqueeze(labels, 1).float()
		x = torch.cat((hr_image, labels), 1)

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

		# c5 = self.conv5(c4)
		# c5 = self.bn5(c5)
		# c5 = self.relu(c5)

		u1 = self.upconv2(c4)
		u2 = self.upconv3(u1 + c3)
		u3 = self.upconv4(u2 + c2)
		u4 = self.upconv5(u3 + c1)
		# u5 = self.upconv5(u4 + c1)

		return u4
