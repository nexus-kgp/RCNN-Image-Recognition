from torchvision import datasets, transforms
import torch
import numpy as np
from PIL import Image

class Preprocess(object):    # the images in the train set are read in PIL format and the resized and converted to Tensors
    """Resize and convert PIL images to to Tensors."""

    def __call__(self, sample):
        image = sample

        image = image.resize((200,200),Image.NEAREST)
        image = np.array(image,dtype='float32')
        image = image.transpose()
        image = torch.from_numpy(image)
        return image
        
data_transform = transforms.Compose([    # the transform class to apply to each of the image being read
        Preprocess()
])

train_tiny_dataset = datasets.ImageFolder(root='./train_image',transform=data_transform)
test_tiny_dataset = datasets.ImageFolder(root='./test_image',transform=data_transform)

img_dataset = {'train' : train_tiny_dataset, 'test': test_tiny_dataset}

train_dataset_loader = torch.utils.data.DataLoader(img_dataset['train'],
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

test_dataset_loader = torch.utils.data.DataLoader(img_dataset['test'],
                                             batch_size=4, shuffle=True,
                                             num_workers=4)

tinyimage_dataloader = {'train':train_dataset_loader, 'test' : test_dataset_loader}
