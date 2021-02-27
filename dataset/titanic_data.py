import torch
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TitanicData(Dataset):
    def __init__(self, split, data_path, isTrain = False):
        super(TitanicData, self).__init__()
        self.isTrain = isTrain
        self.data_path = data_path
        self.split = split
        self.data, self.labels = self.get_data()
        print('Total {} passengers in {} data'.format(len(self.data[:,0]), split))
    
    def get_data(self):
        data_path = self.data_path
        if(self.split == 'train' or self.split == 'val'):
            train = pd.read_csv(data_path, keep_default_na=False)
            # print(train.isna().sum())
            # print(train["Pclass"].unique())
            # print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
            train.drop("Name", axis=1, inplace=True)
            train.drop("Age", axis=1, inplace=True)
            train = train.assign(sex_class = train['Sex'] + "_" + train['Pclass'].astype("str"))
            train["Sex"] = train["Sex"].map({"female":0, "male":1})
            train["sex_class"] = train["sex_class"].map({"female_1":0, "female_2":1, "female_3":2, "male_1":4, "male_2":5, "male_3":6})
            train["fsize"] = train["SibSp"] + train["Parch"] + 1
            train.drop("Ticket", axis=1, inplace=True)
            train.drop("Cabin", axis=1,inplace=True)
            train["Embarked"] = train["Embarked"].map({"S":1, "Q":2, "C":3, "":0})
            print(train)
            # train.drop("Embarked", axis=1, inplace=True)
            train.drop("PassengerId", axis=1, inplace = True)
            survived = train["Survived"]
            # print(survived)
            train.drop("Survived", axis = 1, inplace= True)
            labels = survived.to_numpy()
            data = train.to_numpy()
            print(data)
        elif(self.split == 'test'): 
            train = pd.read_csv(data_path)
            # print(train.isna().sum())
            # print(train["Pclass"].unique())
            # print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
            train.drop("Name", axis=1, inplace=True)
            train.drop("Age", axis=1, inplace=True)
            train = train.assign(sex_class = train['Sex'] + "_" + train['Pclass'].astype("str"))
            train["Sex"] = train["Sex"].map({"female":0, "male":1})
            train["sex_class"] = train["sex_class"].map({"female_1":0, "female_2":1, "female_3":2, "male_1":4, "male_2":5, "male_3":6})
            train["fsize"] = train["SibSp"] + train["Parch"] + 1
            train.drop("Ticket", axis=1, inplace=True)
            train.drop("Cabin", axis=1,inplace=True)
            train["Embarked"] = train["Embarked"].map({"S":1, "Q":2, "C":3, "":0})
            print(train)
            # train.drop("Embarked", axis=1, inplace=True)
            train.drop("PassengerId", axis=1, inplace = True)
            data = train.to_numpy()
            labels = np.zeros((len(data[:,0])))        
        
        # print(len(data[:, 0]))
        # print(data[190], labels[190])
        # print(labels.shape, data.shape)
        return data, labels

    def __len__(self):
        return (len(self.data[:,0]))
    
    def __getitem__(self, index):
        data,label = self.data[index], self.labels[index]
        # print(index, data, label)
        return torch.from_numpy(np.asarray(data)), torch.from_numpy(np.asarray(label))


if __name__ == '__main__':
    titanic_train_data = TitanicData('train', data_path='./dataset/train.csv', isTrain=True)
    # print(titanic_train_data)
    data_loader = DataLoader(titanic_train_data, batch_size = 1, shuffle=False, num_workers = 0)
    for idx, item in enumerate(data_loader):
        data, label = item
        if(idx == 61):
            print(data)
            print(label)
            exit(0)
    # print(titanic_train_data)