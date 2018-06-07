from torchvision import datasets, transforms
import torch

batch_size = 500

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     # transforms.Normalize((0.1307,), (0.3081,))])

trainset = datasets.CIFAR10(root='./data_cifar10', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data_cifar10', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                         shuffle=False, num_workers=2)


dataloader = {'train':trainloader, 'test' : testloader}

test_dataset_len = len(testset)
train_dataset_len = len(trainset)