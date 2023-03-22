import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# from torch implementation of ResNet arhitecture 
#https://github.com/pytorch/vision/blob/a9a8220e0bcb4ce66a733f8c03a1c2f6c68d22cb/torchvision/models/resnet.py#L56-L72
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dropout: float = 0.7):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride) # 
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        self.dropout = nn.Dropout2d(p=dropout)

        self.downsample = downsample
        self.stride = stride
         

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# from torch implementation of ResNet arhitecture 
#https://github.com/pytorch/vision/blob/a9a8220e0bcb4ce66a733f8c03a1c2f6c68d22cb/torchvision/models/resnet.py#L56-L72
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dropout: float = 0.7):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(p=dropout)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

   # set up variables
        self.basic_block = BasicBlock
        self.bottleneck = Bottleneck
        self.layers = [2,2]
        self.norm_layer = nn.BatchNorm2d
        self.inplanes = 64 
        self.dropout = dropout

        # first layer taking 3*240*240 as input and return 64*56*56
        #https://arxiv.org/pdf/1512.03385.pdf
        self.conv1 = self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2d = nn.Dropout2d(p=dropout)

        self.layer1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.dropout2d
        )

        self.layer2 = self._make_layer(block=BasicBlock, planes=64, blocks=1, stride=1)
        self.layer3 = self._make_layer(block=Bottleneck, planes=32, blocks=1, stride=2)
        self.layer4 = self._make_layer(block=BasicBlock, planes=256, blocks=1, stride=1)
        self.layer5 = self._make_layer(block=Bottleneck, planes=128, blocks=1, stride=2)
        self.layer6 = self._make_layer(block=BasicBlock, planes=512, blocks=1, stride=1)
        self.layer7 = self._make_layer(block=Bottleneck, planes=128, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        norm_layer = self.norm_layer

        #downsample using conv1x1
        if stride !=1 or self.inplanes != planes*block.expansion: 
            downsample = nn.Sequential(
                conv1x1(in_planes=self.inplanes, out_planes=planes*block.expansion, stride=stride),
                norm_layer(planes*block.expansion)
            )

        layers = list()
        layers.append(block( 
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride, 
                    downsample=downsample,
                    norm_layer=norm_layer,
                    dropout=self.dropout
                    )
                )
        self.inplanes = planes*block.expansion 

        #if we have mnore than 1 block 
        for _ in range(1, blocks):
            layers.append(block(
                        inplanes=self.inplanes,
                        planes=planes, 
                        #stride=stride, # no need for strade , styrade=1 by default avoid downsample 2x
                        #downsample=downsample,
                        norm_layer=norm_layer,
                        dropout=self.dropout
                        )
                    )
            
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
