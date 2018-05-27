import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model import RCNN
from dataloader import tinyimage_dataloader

net = RCNN()

learning_rate = 0.001
epoch = 1

criterion = nn.BCELoss(size_average=False)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 'min' ,
#     factor=0.1 ,
#     patience=10,
#     verbose=True)


### Tiny Image Dataset

loss_trend = []

phase = 'train'

for i in range(epoch):
	loss_each_epoch = []
	for images,labels in tinyimage_dataloader['train']:
		images = Variable(images)
		target = torch.zeros(labels.shape[0],2)

		## OHE
		for i in range(4):
			target[i][labels[i]] = 1

		output = net(images)
		print(output)
		print(target)

		loss = criterion(output, Variable(target))
		loss.backward()
		optimizer.step()

		print(loss)
		loss_each_epoch.append(loss.view(1).data.numpy()[0])

	loss_trend.append(sum(loss_each_epoch))



## Testing
# label = torch.ones((1,5))
# test = torch.ones(3*64*64).view(1,3,64,64)
# out = net(test)
# loss = criterion(out,label)
# print(loss)
# loss.backward()
# scheduler.step(loss)


print(net)
print("k..thnx..bye")