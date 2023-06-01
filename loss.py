from torch import nn

def Prediction_loss(outputs, labels):
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_func(outputs, labels.float())
    return loss

