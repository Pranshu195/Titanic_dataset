import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm



def eval_test(model, dataset, criterion):
    sub_file = open("./dataset/submisission.csv", 'w+')
    sub_file.write("PassengerId,Survived\n")
    pessengerId = 892
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    correct = 0.0
    total = 0.0
    val_loss = 0.0
    for i , (data, labels) in enumerate(tqdm(test_loader)):
        inputs = Variable(data).float()
        # print('input size ' ,inputs)
        # labels = Variable(labels)
        output = model(inputs)
        output = output.view(-1)
        # print('outsize {} , type {}'.format(output.shape, labels.shape))
        # print('out = ', output, labels)
        # loss = criterion(output, labels)
        # val_loss += loss.item()
        if(output > 0.5):
            pred = 1
        else:
            pred = 0
        sub_file.write(str(pessengerId) + ',' + str(pred) + '\n')
        pessengerId += 1
