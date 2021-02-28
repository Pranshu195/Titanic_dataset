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
            train = train.fillna(0)
            # train.drop("Embarked", axis=1, inplace=True)
            train.drop("PassengerId", axis=1, inplace = True)
            survived = train["Survived"]
            # print(survived)
            train.drop("Survived", axis = 1, inplace= True)
            labels = survived.to_numpy()
            data = train.to_numpy()
            data = self.get_data_pca(data)
            # print(data)
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
            train["Embarked"] = train["Embarked"].map({"S":1, "Q":2, "C":3})
            train = train.fillna(0)
            # print("Here")
            # print(train)
            # train.drop("Embarked", axis=1, inplace=True)
            train.drop("PassengerId", axis=1, inplace = True)
            data = train.to_numpy()
            labels = np.zeros((len(data[:,0])))
            data = self.get_data_pca(data)      
        
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

    def get_data_pca(self, data):
        ######Standardize the data
        mean = np.zeros((len(data[0, :])), dtype=np.float)
        std = np.zeros((len(data[0, :])), dtype=np.float)
        data_std = np.zeros((data.shape), dtype=np.float)
        for i in range(len(data[0, :])):
            mean[i] = np.mean(data[:, i])
            # print("mean = {}".format(mean[i]))
            std[i] = np.std(data[:, i])
            # print("std = {}".format(std[i]))
            data_std[:, i] = (data[:, i] - mean[i]) / std[i]
        # print(data_std)
       
        ##### covariance matrix
        cov_mat = np.cov(data_std.T)
        # print(cov_mat)

        #####Eigen decomposition of the covirance matrix
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
        # print("Eigen vals = {}".format(eigen_values))
        # print("Eigen vactors = {}".format(eigen_vectors))

        #### Calculating the explained variance of each component
        variance_explained = []
        for i in eigen_values:
            variance_explained.append((i/sum(eigen_values))*100)
        # print(variance_explained)
        
        ###  using first 6 components as they explain 100% of the dataset
        projection_matrix = (eigen_vectors.T[:][:5]).T
        # print(projection_matrix)

        #### Getting the product of original std data and projection matrix
        data_pca = data_std.dot(projection_matrix)
        # print(data_pca)
        return data_pca



##Testing the dataloader
if __name__ == '__main__':
    titanic_train_data = TitanicData('train', data_path='./dataset/train.csv', isTrain=False)
    # print(titanic_train_data)
    data_loader = DataLoader(titanic_train_data, batch_size = 1, shuffle=False, num_workers = 0)
    for idx, item in enumerate(data_loader):
        data, label = item
        if(idx == 61):
            print(data)
            print(label)
            exit(0)
    # print(titanic_train_data)