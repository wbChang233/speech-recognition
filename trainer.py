import torch
from tqdm import tqdm


def train(train_loader, model, criterion, optimizer, epoch):
    print('Epoch: %d' % (epoch + 1))
    model.train()
    sum_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to('cuda')  # 将输入数据转移到cuda设备上
        labels = labels.to('cuda')  # 将标签数据转移到cuda设备上
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss

def valid(valid_loader, model):
    Correct_top1 = 0
    batchSum = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            batchSum += len(inputs)
            inputs = inputs.to('cuda')  # 将输入数据转移到cuda设备上
            labels = labels.to('cuda')
            outputs = model(inputs)
            # 计算top1分类精度
            Predictions_top1 = outputs.argmax(dim=1)
            target1 = labels.argmax(dim=1)
            Correct_top1 += (Predictions_top1 == target1).sum().item()  # 累加正确预测的数量

    return Correct_top1 / batchSum * 100
