# Adversarial Robustness - Theory and Practice
# Chapter 1 - Introduction to adversarial robustness
# https://adversarial-ml-tutorial.org/introduction/
import math
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

from targeted_attack import targeted_attack
from non_targeted_attack import non_targeted_attack


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


if __name__ == '__main__':
    # 341 is the class index corresponding to "hog"
    true_class = 341
    # 404 is the class index corresponding to "airliner"
    target_class = 404

    # read the image
    # resize to 244
    # convert to PyTorch Tensor
    pig_img = Image.open('pig.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # inputs to modules should be of the form batch_size x num_channels x height x width
    pig_tensor = preprocess(pig_img)[None, :, :, :]

    # plot image (note that numpy using HWC whereas PyTorch user CHW, so we need to convert)
    # plt.imshow(pig_tensor[0].numpy().transpose(1, 2, 0))
    # plt.show()

    # values are standard normalization for ImageNet images,
    # from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # load pre-trained ResNet50,
    model = resnet50(pretrained=True)
    # and put into evaluation mode (necessary to e.g. turn off batchnorm)
    model.eval()

    # from predictions
    # pred is a 1000 dimensional vector containing the class logits for the 1000 ImageNet classes
    pred = model(norm(pig_tensor))

    # decode predication to text
    with open('/home/xcy/.keras/models/imagenet_class_index.json') as f:
        imagenet_class = {int(i): x[1] for i, x in json.load(f).items()}
    print(imagenet_class[pred.max(dim=1)[1].item()])

    loss_value = nn.CrossEntropyLoss()(model(norm(pig_tensor)), torch.LongTensor([true_class]))
    print('loss: {}'.format(loss_value.item()))
    print('probability: {}'.format(math.exp(-loss_value.item())))

    # creating an adversarial example by non-targeted attack
    delta1 = non_targeted_attack(model, norm, pig_tensor, true_class)
    pred = model(norm(pig_tensor + delta1))
    print('True class probability: ', nn.Softmax(dim=1)(pred)[0, true_class].item())
    max_class = pred.max(dim=1)[1].item()
    print('Predicted class: ', imagenet_class[max_class])
    print('Predicted probability: ', nn.Softmax(dim=1)(pred)[0, max_class].item())
    # plt.imshow((pig_tensor + delta1)[0].detach().numpy().transpose(1, 2, 0))

    # creating an adversarial example by targeted attack
    delta2 = targeted_attack(model, norm, pig_tensor, true_class, target_class)
    pred = model(norm(pig_tensor + delta2))
    print('True class probability: ', nn.Softmax(dim=1)(pred)[0, true_class].item())
    max_class = pred.max(dim=1)[1].item()
    print('Predicted class: ', imagenet_class[max_class])
    print('Predicted probability: ', nn.Softmax(dim=1)(pred)[0, max_class].item())
    # plt.imshow((pig_tensor + delta2)[0].detach().numpy().transpose(1, 2, 0))

    # show delta
    # zoomed in by a factor of 50 because it would be impossible to see otherwise.
    # plt.imshow((50 * delta + 0.5)[0].detach().numpy().transpose(1, 2, 0))
    # plt.show()
