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
		
	def __call__(self, obs):
		'''
		Your code here
		'''
		
class Model(nn.Module):
	def __init__(self):
		super().__init__()
		
		'''
		Your code here
		'''
		# The number of sentiment classes
		self.target_size = 6
		self.width=100

		# The Dropout Layer Probability. Same for all layers
		self.dropout_prob = 0.0

		# Option to use a stacked LSTM
		self.num_lstm_layers = 1

		# Option to Use a bidirectional LSTM

		self.isBidirectional = False

		if self.isBidirectional:
			self.num_directions = 2
		else:
			self.num_directions = 1

		# The Number of Hidden Dimensions in the LSTM Layers
		self.hidden_dim = 32

		ks = 5
		self.conv1 = nn.Conv2d(3 , 16 , ks, 2)
		self.conv2 = nn.Conv2d(16, 32 , ks, 2)
		self.conv3 = nn.Conv2d(32, 64 , ks, 2)
		self.conv4 = nn.Conv2d(64, 128, ks, 2)

		self.lstm_layer = nn.LSTM(
				input_size = 128,
				hidden_size = self.hidden_dim,
				num_layers = self.num_lstm_layers,
				bidirectional = self.isBidirectional,
				batch_first = True
			)

		self.relu = nn.LeakyReLU(inplace=True)
		self.linear = nn.Linear(32, 6)



		
	def forward(self, hist):
		'''
		Your code here
		Input size: (batch_size, sequence_length, channels, height, width)
		Output size: (batch_size, sequence_length, 6)
		'''

		batch_size = hist.shape[0]
		sequence_length = hist.shape[1]
		x = hist

		x = x.view(batch_size * sequence_length, 3, 64, 64)
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))

		x = x.view(batch_size, sequence_length, -1)
		x, hidden = self.lstm_layer(x)
		x = self.linear(x)

		return x

		
	def policy(self):
		return Policy(self)
