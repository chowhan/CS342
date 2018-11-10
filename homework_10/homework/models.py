from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Policy:
	"""
	Class used for evaluation. Will take a single observation as input in the __call__ function and need to output the l6 dimensional logits for next action
	"""
	def __init__(self, model):
		'''
		Your code here
		'''
		self.model = model
		self.hist = []
		
	def __call__(self, obs):
		'''
		Your code here
		'''
		# obs = obs.contiguous().view(1, 1, -1)
		# out, self.hidden = self.model(obs, True, self.hid)
		# return out[0, -1, :]
		self.hist.append(obs)
		if len(self.hist) > self.model.width:
			self.hist = self.hist[-self.model.width:]
		x = torch.stack(self.hist, dim=0)[None]
		return self.model(x)[0,-1,:]
		
class Model(nn.Module):
	def __init__(self, channels=[16,32,16], ks=5):
		super().__init__()
		
		'''
		Your code here
		'''

		ks = 5
		self.conv1 = nn.Conv2d(3 , 16, ks, 2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, ks, 4, 1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, ks, 2)
		self.bn3 = nn.BatchNorm2d(64)

		self.linear1 = nn.Linear(256, 128)
		self.linear2 = nn.Linear(128, 32)

		self.relu = nn.ReLU(inplace=True)

		# self.dropout_layer = nn.Dropout(self.dropout_prob)

		c0 = 32
		layers = []
		self.width = 1
		for c in channels:
			layers.append( nn.Conv1d(c0, c, ks) )
			layers.append( nn.LeakyReLU() )
			c0 = c
			self.width += ks-1
		layers.append( nn.Conv1d(c0, 6, ks) )
		self.width += ks-1
		
		self.model = nn.Sequential(*layers)



		
	def forward(self, hist, test = False, hidden = None):
		'''
		Your code here
		Input size: (batch_size, sequence_length, channels, height, width)
		Output size: (batch_size, sequence_length, 6)
		'''

		batch_size = hist.shape[0]
		sequence_length = hist.shape[1]
		x = hist

		x = x.view(batch_size * sequence_length, 3, 64, 64)
		x = self.bn1(self.relu(self.conv1(x)))
		x = self.bn2(self.relu(self.conv2(x)))
		x = self.bn3(self.relu(self.conv3(x)))

		x = x.view(batch_size, sequence_length, -1)
		x = self.relu(self.linear1(x))
		x = self.relu(self.linear2(x))
		x = x.permute(0, 2, 1)
		x = F.pad(x, (self.width-1,0))
		x = self.model(x)
		x = x.permute(0, 2, 1)

		out = x
		if test:
			return out, hidden
		else:
			return out
		
	def policy(self):
		return Policy(self)
