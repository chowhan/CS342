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

		self.conv1 = nn.Conv2d(9, 128, 5, 2, 2)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv2 = nn.Conv2d(128, 256, 5, 2, 2)
		self.bn3 = nn.BatchNorm2d(256)

		self.upconv2 = nn.ConvTranspose2d(259, 128, 5, 2, 2, 1)
		self.bn4 = nn.BatchNorm2d(128)
		self.upconv3 = nn.ConvTranspose2d(128, 3, 5, 2, 2, 1)

		nn.init.constant_(self.upconv3.weight, 0)
		nn.init.constant_(self.upconv3.bias, 0)

		self.relu = nn.LeakyReLU(inplace=True)


	def forward(self, image, labels):
		
		'''
		Your code here
		'''
		hr_image = nn.functional.interpolate(image, scale_factor=4, mode='nearest')
		labels = one_hot(labels)
		#labels = torch.unsqueeze(labels, 1).float()
		x = torch.cat((hr_image, labels), 1)

		# print(labels.size())
		#x = one_hot(labels)
		c1 = self.conv1(x)
		c1 = self.bn2(c1)
		c1 = self.relu(c1)

		c2 = self.conv2(c1)
		c2 = self.bn3(c2)
		c2 = self.relu(c2)

		c2 = torch.cat((image, c2), 1)

		u2 = self.upconv2(c2)
		#u2 = self.bn4(u2)
		#u2 = self.relu(u2)
		u3 = self.upconv3(u2 + c1)
		#u3 = self.bn3(u3)
		#u3 = self.relu(u3)


		return u3
