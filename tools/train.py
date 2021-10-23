import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dataset.titanic_data import TitanicData
from model.titanic_regression import TitanicModel
from model.random_forest import Feature_Selection_Node, Decision_Node
from tools.evaluate import evaluate
from tools.eval_test import eval_test
from tqdm import tqdm
import numpy as np
import random
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


torch.autograd.set_detect_anomaly(True)

# def frequency(d):
#     dic = {}
#     for item in d:
#         if item in dic.keys():
#             dic[item] = dic[item] + 1
#         else:
#             dic[item] = 1
#     dic = {"values" : dic.keys(), "count": dic.values()}
#     df = pd.DataFrame.from_dict(dic, orient = 'index').transpose().sort_values(["values"])
#     df["cum"] = df["count"] / df["count"].sum()
#     value = df["cum"].values
#     value = torch.from_numpy(value).float()
#     value = 1- value*value
#     value = value.sum(-1)
#     return value

def train(data_path, train):
    batch_size = 2
    train_dataset = TitanicData('train', data_path)
    val_dataset = TitanicData('val', data_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # mask = Feature_Selection_Node(100, batch_size)
    # decision = Decision_Node(100, 200, 1, batch_size)
    # params = list(mask.parameters()) + list(decision.parameters())
    model = TitanicModel(22, 15, 10, 1)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1E-4)
    optimizer = torch.optim.Adam( model.parameters(), lr=1e-3)

    #optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    # optimizer.add_param_group({'params': model.parameters(), 'lr': 1e-4})
    
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    epochs = 30
    best_val_acc = None
    best_val_loss = None
    best_epoch = -1
    if(train == True):
        with torch.no_grad():
            best_val_acc, best_val_loss = evaluate(model, val_dataset, criterion)
            print('Best Validation Accuracy : {}'.format(best_val_acc))
            print('Best Validation Loss : {}'.format(best_val_loss))
            # exit(0)
        for epoch in range(epochs):
            train_loss = 0.0
            model.train()
            for i, (data, labels) in enumerate(tqdm(train_loader)):
                inputs = Variable(data).float()
                labels = Variable(labels).float()
                optimizer.zero_grad()
                output = model(inputs)
                output = output.view(-1)
                loss = criterion(output, labels)
                train_loss += loss.item()
                # print('before ', i, inputs, output)
                loss.backward()
                
                optimizer.step()
                # print('After ', list(model.parameters())[0])          
                #exit(0)
                if i % 500 == 0:
                    print('Epoch {}/{} : Step {}/{}, Loss: {:.4f}'
                            .format(epoch + 1, 30, i + 1, len(train_loader), loss.item()))
            with torch.no_grad():
                validation_acc, val_loss = evaluate(model, val_dataset, criterion)
            model.train()
            scheduler.step()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = validation_acc
                best_epoch = epoch
                # torch.save(model.state_dict(), "Trained_models_dump/titanic_best_epoch_fastlr_improved_data_pca_{}.pth".format(epoch + 1))
            print('Best Validation Loss : {}'.format(best_val_loss))
            print('Best Validation Accuracy : {}'.format(best_val_acc))
            print('Best Epoch: {}'.format(best_epoch + 1))
            print('Epoch {}/{} Done | Train Loss : {:.4f} | Validation Loss : {:.4f} | Validation Accuracy : {:.4f}'
                .format(epoch + 1, 30, train_loss / len(train_loader), val_loss, validation_acc))
    else:
        print("for creating submission file")
        model.load_state_dict(torch.load('Trained_models_dump/titanic_best_epoch_fastlr_improved_data_pca_30.pth'))
        dataset1 = TitanicData('test', "./dataset/test.csv", isTrain=False)
        eval_test(model, dataset1, criterion)
    return best_val_loss

if __name__ == '__main__':
    torch.manual_seed(1111)
    np.random.seed(1111)
    random.seed(1111)
    best_val_loss = train("./dataset/train.csv", True)
    print(best_val_loss)