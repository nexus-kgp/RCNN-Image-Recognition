import torch
import torch.nn as nn
import torch.nn.functional as F

### LRN ###
LRN_alpha = 1e-3
LRN_size = 5
#LRN_N = K/8 + 1  ?? figure this out
###########

### Dropout ###
DRPT_prob = 0.5
###############

c1_out_channels = K = 96  ## AKA K
no_of_RCL_blocks = 4

class RCLBlock(nn.Module):
	def __init__(self):
		super(RCLBlock, self).__init__()

		# self.LRN = nn.LocalResponseNorm(LRN_size,alpha=LRN_alpha)
		self.feedforward_filter = nn.Conv2d(K,K,3,padding=1)
		self.recurrent_filter = nn.Conv2d(K,K,3,padding=1)
		self.timesteps = 6

	def forward(self, input):
		out = feedforward_output = (F.relu(self.feedforward_filter(input)))
		for i in range(self.timesteps):
			out = self.recurrent_filter(out)
			out = out + feedforward_output
			out = (F.relu(out))
		return out


class RCNN(nn.Module):
	def __init__(self, n_classes=2):
		super(RCNN, self).__init__()
		
		self.LRN = nn.LocalResponseNorm(LRN_size,alpha=LRN_alpha)
		self.Dropout = nn.Dropout(DRPT_prob)
		self.MaxPool = nn.MaxPool2d(3,stride=2,padding=1)
		self.conv1 = nn.Conv2d(1,K,5)
		self.ReLU = nn.ReLU()

		self.RCL1 = RCLBlock()
		self.RCL2 = RCLBlock()
		self.RCL3 = RCLBlock()
		self.RCL4 = RCLBlock()
		self.RCL5 = RCLBlock()
		self.RCL6 = RCLBlock()

		self.Linear = nn.Linear(K,n_classes)

	


	def forward(self, x):
		out = self.conv1(x)
		# return out
		out = self.MaxPool(out)
		# return out
		 ## RCL Block :- 1
		out = self.RCL1(out)
		out = self.Dropout(out)
		 ## RCL Block :- 2
		out = self.RCL2(out)
		out = self.MaxPool(out)

		out = self.Dropout(out)
		
		## RCL Block :- 3
		out = self.RCL3(out)
		out = self.Dropout(out)
		 ## RCL Block :- 4
		out = self.RCL4(out)

		out = self.MaxPool(out)

		out = self.Dropout(out)
		 ## RCL Block :- 4
		out = self.RCL5(out)
		out = self.Dropout(out)
		 ## RCL Block :- 4
		out = self.RCL6(out)

		## Global Max Pooling
		out = F.max_pool2d(out, out.size()[2:])  ## after this, out.shape == N,K,1,1
		out = out.view(out.size()[0],out.size()[1])  ##  after this, out.shape == N,K  
		out = self.Linear(out)  ## after this, out.shape == N,n_classes
		out = F.softmax(out,dim=1)
		return out
