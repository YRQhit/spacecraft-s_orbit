import torch
import torch.nn as nn
from torchvision import models

class CustomResNet(nn.Module):
    def __init__(self, input_dim=6, output_dim=3, pretrained=True):
        super(CustomResNet, self).__init__()

        # 加载预训练的 ResNet 模型
        self.resnet = models.resnet18(pretrained=pretrained)

        # 冻结 ResNet 的参数（可选）
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 修改 ResNet 第一层的输入通道数为 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # 获取 ResNet 最后一层的输出特征维度
        resnet_out_dim = self.resnet.fc.in_features

        # 添加自定义的全连接层
        self.custom_fc = nn.Sequential(
            nn.Linear(resnet_out_dim, output_dim)
        )

    def forward(self, x):
        # 将一维数据转换为类似图像的形式
        x = x.view(1, 1, -1, 1)  # 假设输入是一维数据，将其视为一个通道的图像

        # 使用 ResNet 模型进行特征提取
        features = self.resnet(x)

        # 将 ResNet 输出传递给自定义全连接层
        output = self.custom_fc(features.view(features.size(0), -1))

        return output

# 创建自定义 ResNet 模型实例
custom_resnet_model = CustomResNet(input_dim=6, output_dim=3, pretrained=True)

# 创建一个随机输入
input_tensor = torch.randn(6)  # 六维的输入

# 将输入传递给模型
output_tensor = custom_resnet_model(input_tensor)
print(output_tensor.shape)
