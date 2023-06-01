import torch.nn as nn

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        # 定义一个2D卷积层，输入通道数为1，输出通道数为40，卷积核大小为3x3，步长为1，填充为1
        self.conv = nn.Conv2d(1, 40, kernel_size=3, stride=1, padding=1)
        # 定义一个LSTM层，输入特征数为10240，隐藏单元数为32，层数为1
        self.lstm = nn.LSTM(10240, 32, num_layers=1)
        # 定义一个全连接层，输入特征数为32，输出特征数为3（对应三种语种）
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        # stft
        # x = x.float()
        # 通过卷积层，提取空间特征
        x = self.conv(x)
        # 将卷积层的输出转换为LSTM层的输入格式（序列长度，批量大小，特征数）
        x = x.permute(2, 0, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        # 通过LSTM层，提取时间特征
        x, _ = self.lstm(x)
        # 取最后一个时间步的输出作为全连接层的输入
        x = x[-1]
        # 通过全连接层，得到最终的输出
        output = self.fc(x)
        return output
