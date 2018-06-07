import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import RCNN
from dataloader_mnist import dataloader,batch_size,test_dataset_len,train_dataset_len

n_classes = 10
net = RCNN(n_classes=n_classes)

learning_rate = 1e-3
epoch = 30

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
# 	optimizer, 'min' ,
# 	factor=0.1 ,
# 	patience=(train_dataset_len/batch_size)*3,
# 	verbose=True)



use_gpu = torch.cuda.is_available()

if use_gpu:
	net = net.cuda()


loss_trend = []
accuracy_trend = []

phase = 'train'

for i in range(epoch):
	loss_each_epoch = []
	running_accuracy = []
	mini_count = 1
	for images,labels in dataloader['train']:
		net = net.train()
		if use_gpu:
			images = images.cuda()
			labels = labels.cuda()

		optimizer.zero_grad()
		# scheduler.optimizer.zero_grad()
		output = net(images)
		# print(output)
		# print(target)

		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		# scheduler.step(loss)
		loss_to_append = loss.clone().cpu().view(1).data.numpy()[0]
		print("Epoch : {}, Mini-Epoch : {}, Loss: {}".format(i+1,mini_count,loss_to_append))
		mini_count += 1
		loss_each_epoch.append(loss_to_append)

	loss_trend.append(sum(loss_each_epoch))

	for images,labels in dataloader['test']:
		net = net.eval()
		if use_gpu:
			images = images.cuda()
			labels = labels.cuda()

		output = net(images)
		predicted_labels = torch.argmax(output, dim=1)

		minibatch_accuracy = torch.eq(predicted_labels,labels).cpu().sum().view(1).numpy()[0]
		running_accuracy.append(minibatch_accuracy)

	accuracy_trend.append( sum(running_accuracy)/test_dataset_len )

	print('##### Epoch {} #####'.format(i+1))
	print('Loss : {}'.format(sum(loss_each_epoch)))
	print('Accuracy : {}'.format( sum(running_accuracy)/test_dataset_len ))
	print('####################')


print(net)
print("k..thnx..bye")
