import torch
from torch import nn
from torch import optim


def non_targeted_attack(model, norm, tensor_to_hack, true_class):
    """
    Non-targeted attack. Create an adversarial example to make model mis-classify.
    :param model: PyTorch pre-trained model
    :param norm: Normalizer
    :param tensor_to_hack: PyTorch Tensor to be hacked
    :param true_class: int: true label of original Tensor
    :return: perturbation
    """
    epsilon = 2. / 255
    delta = torch.zeros_like(tensor_to_hack, requires_grad=True)
    opt = optim.SGD([delta], lr=1e-1)

    for t in range(30):
        pred = model(norm(tensor_to_hack + delta))
        # just maximize the loss of the correct class
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([true_class]))

        if t % 10 == 0:
            print(t, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        # simply clipping the values that exceed ϵ magnitude to ±ϵ
        delta.data.clamp_(-epsilon, epsilon)

    return delta
