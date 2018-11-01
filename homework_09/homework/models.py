from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

def get_seq_mask(input_seq_lens, max_seq_len):
    return torch.as_tensor(np.asarray([[1 if j < input_seq_lens.data[i].item() else 0 for j in range(0, max_seq_len)] for i in range(0, input_seq_lens.shape[0])]), dtype=torch.float)#.cuda()


class SeqPredictor:

	"""
	Helper class to use within the SeqModel class to make sequential predictions.
	Not mandatory to use the class.
	However, this class can be helpful since the inputs are different during training and evaluation.
	During training, the whole sentence is passed at once to the network, while during evaluation the sequence is passed one action at a time.
	"""

	def __init__(self, model):
		self.model = model
		self.hidden = None

	def __call__(self, input):
		"""
		@param input: A single input of shape (6,) indicator values (float: 0 or 1)
		@return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty)
		"""
		input = input.view(1, 6, 1)
		output, self.hidden = self.model(input, self.hidden, True)
		return output[0,:,-1]

class SeqModel(nn.Module):

	"""
	Define your recurrent neural network here
	"""
	def __init__(self):
		
		super().__init__()
		self.hsize = 32
		self.rnn = nn.LSTM(6, self.hsize, 2)
		self.l1 = nn.Linear(self.hsize, 6)

	def forward(self, input, hidden=None, from_test=False):
		"""
		IMPORTANT: Do not change the function signature of the forward() function unless the grader won't work.
		@param input: A sequence of input actions (batch_size x 6 x sequence_length)
		@return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty). Shape: batch_size x 6 x sequence_length
		"""
		inp = input.permute(2, 0, 1)
		out, hidden = self.rnn(inp, hidden)
		out = self.l1(out)
		out = out.permute(1, 2, 0)
		if from_test:
			return out, hidden
		else:
			return out

	def predictor(self):
		return SeqPredictor(self)



##########################################
# Simple example of a constant predictor #
##########################################

class SimpleSeqPredictor:
	def __init__(self, model):
		self.model = model

	def __call__(self, input):
		return self.model(input[None,:,None])[0,:,-1]

class ConstantModel(nn.Module):
	"""
	A constant prediction
	"""
	def __init__(self):
		super().__init__()
		self.alpha = nn.Parameter(torch.ones(6))

	def forward(self, input):
		return self.alpha[None,:,None] * (0*input+1)

	def predictor(self):
		return SimpleSeqPredictor(self)
