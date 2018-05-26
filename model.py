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

c1_out_channels = K = 10  ## AKA K
no_of_RCL_blocks = 4

class RCNN(nn.Module):
	def __init__(self, n_classes=5):
		super(RCNN, self).__init__()
		
		self.LRN = nn.LocalResponeNorm(LRN_size,alpha=LRN_alpha)
		self.Dropout = nn.Dropout(DRPT_prob)
		self.MaxPool = nn.MaxPool2d(3,2)
		self.conv1 = nn.Conv2d(3,K,5)
		self.ReLU = nn.ReLU()
		self.RCL_Convs = [[nn.Conv2d(K,K,3)]*2 for i in range(no_of_RCL_blocks)]
		self.Linear = nn.Linear(K/2,n_classes)   ## Verify that after second maxpool K goes to K/2

	def RCLBlock(self,input,index):
		feedforward_filter =  self.RCL_Convs[index][0]
		recurrent_filter = self.RCL_Convs[index][1]
		## t = 0
		out = feedforward_output = feedforward_filter(input)
		out = self.ReLU(out)
		out = self.LRN(out)
		## t = 1 to 3
		for i in range(3):
			out = recurrent_filter(out)
			out = out + feedforward_output
			out = self.ReLU(out)
			out = self.LRN(out)

		return out

	def forward(self, x):
		out = self.conv1(x)
		out = self.MaxPool(out)
		 ## RCL Block :- 1
		out = RCLBlock(self,out,0)
		out = self.Dropout(out)
		 ## RCL Block :- 2
		out = RCLBlock(self,out,1)
		out = self.MaxPool(out)

		out = self.Dropout(out)
		
		## RCL Block :- 3
		out = RCLBlock(self,out,2)
		out = self.Dropout(out)
		 ## RCL Block :- 4
		out = RCLBlock(self,out,3)

		## Global Max Pooling
		out = F.max_pool2d(out, out.size()[2:])  ## after this, out.shape == N,K,1,1
		out = out.view(out.size()[0],out.size()[1])  ##  after this, out.shape == N,K  
		out  = self.Linear(out)  ## after this, out.shape == N,n_classes
		out = F.softmax(out)
		return out
