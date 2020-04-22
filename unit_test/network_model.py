import torch
from Regression.network.model import VGG19, ResNet18, ResNet34, ResNet50
from Regression.config import config


image_size = config.generate.image_size
batch_size = config.network.batch_size
label_num = len(config.dataset.labels)


def run_model():
    run_vgg19()
    # run_resnet18()
    # run_resnet34()
    # run_resnet50()

    print('Models Ok!')


def run_vgg19():
    vgg19 = VGG19()
    x = torch.zeros((batch_size, 1, image_size, image_size))
    output = vgg19(x)
    print(vgg19.features)
    assert output.size()[0] == batch_size and output.size()[1] == label_num


def run_resnet18():
    resnet18 = ResNet18()
    print(resnet18)
    x = torch.zeros((batch_size, 1, image_size, image_size))
    output = resnet18(x)

    assert output.size()[0] == batch_size and output.size()[1] == label_num


def run_resnet34():
    resnet34 = ResNet34()
    print(resnet34)
    x = torch.zeros((batch_size, 1, image_size, image_size))
    output = resnet34(x)

    assert output.size()[0] == batch_size and output.size()[1] == label_num


def run_resnet50():
    resnet50 = ResNet50()
    print(resnet50)
    x = torch.zeros((batch_size, 1, image_size, image_size))
    output = resnet50(x)

    assert output.size()[0] == batch_size and output.size()[1] == label_num


def run_densenet121():
    pass
