import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(model, dataset, criterion):
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    model.eval()
    correct = 0.0
    total = 0.0
    val_loss = 0.0
    for i , (data, labels) in enumerate(tqdm(val_loader)):
        inputs = Variable(data).float()
        # print('input size ' ,inputs)
        labels = Variable(labels)
        output = model(inputs)
        output = output.view(-1)
        # print('outsize {} , type {}'.format(output.shape, labels.shape))
        # print('out = ', output, labels)
        loss = criterion(output, labels)
        val_loss += loss.item()
        if(output > 0.5):
            pred = 1
        else:
            pred = 0
        if(labels == pred):
            correct += 1
        total += 1

    return correct/total, val_loss/(len(val_loader))
