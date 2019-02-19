import torch
from torch import nn
from torch import optim


def targeted_attack(model, norm, tensor_to_hack, true_class, target_class):
    """
    Targeted attack. Create an adversarial example to make model mis-classify as target class.
    :param model: PyTorch pre-trained model
    :param norm: Normalizer
    :param tensor_to_hack: PyTorch Tensor to be hacked
    :param true_class: int: true label of original Tensor
    :param target_class: int: label of attack target
    :return: perturbation
    """
    epsilon = 2. / 255
    delta = torch.zeros_like(tensor_to_hack, requires_grad=True)
    opt = optim.SGD([delta], lr=5e-3)

    for t in range(100):
        pred = model(norm(tensor_to_hack + delta))
        # maximize the loss of the correct class while also minimizing the loss of the target class.
        loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([true_class])) +
                nn.CrossEntropyLoss()(pred, torch.LongTensor([target_class])))

        if t % 10 == 0:
            print(t, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        # simply clipping the values that exceed ϵ magnitude to ±ϵ
        delta.data.clamp_(-epsilon, epsilon)

    return delta
