import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import RCNN

net = RCNN()

learning_rate = 0.001

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min' ,
    factor=0.5 ,
    patience=1000 ,
    verbose=True)


## Testing
label = torch.ones((1,5))
test = torch.ones(3*64*64).view(1,3,64,64)
out = net(test)
loss = criterion(out,label)
print(loss)
loss.backward()
scheduler.step(loss)


print(net)
print("k..thnx..bye")