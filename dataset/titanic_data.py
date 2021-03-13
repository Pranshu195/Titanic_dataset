import torch
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TitanicData(Dataset):
    def __init__(self, split, data_path):
        super(TitanicData, self).__init__()
        self.data_path = data_path
        self.split = split
        self.data, self.labels = self.get_data()
        print('Total {} passengers in {} data'.format(len(self.data[:,0]), split))
    
    def data_util(self):
        data_path = self.data_path
        df = pd.read_csv(data_path)
        # train = train[:50]
        df = self.process_family(df)
        df = self.process_embarked(df)
        df = self.process_cabin(df)
        df = self.get_titles(df)
        df = self.get_age(df)
        df = self.process_names(df)
        df = df.assign(sex_class = df['Sex'] + "_" + df['Pclass'].astype("str"))
        df["Sex"] = df["Sex"].map({"female":0, "male":1})
        df["sex_class"] = df["sex_class"].map({"female_1":0, "female_2":1, "female_3":2, "male_1":4, "male_2":5, "male_3":6})
        df.drop("Ticket", axis=1, inplace=True)
        df.drop("PassengerId", axis=1, inplace = True)
        return df
        # survived = df["Survived"]
        # df.drop("Survived", axis = 1, inplace= True)

        # train = train[:60]
        # print(train.shape)
    
        # labels = survived.to_numpy()
        # data = train.to_numpy()
        # data = self.get_data_pca(data)
        # data = data[:60]
        # print(data[:5])
        # print(data.shape)


    def get_data(self):
        data_path = self.data_path
        df = self.data_util()
        if self.split == 'train':
            survived = df["Survived"]
            df.drop("Survived", axis = 1, inplace= True)
            labels = survived.to_numpy()
            data = df.to_numpy()
            data = self.get_data_pca(data)
            data = data[60:]
            survived = survived[60:]
            print(data.shape)
            # print(data[:5])
        elif self.split == 'val' :
            survived = df["Survived"]
            df.drop("Survived", axis = 1, inplace= True)
            labels = survived.to_numpy()
            data = df.to_numpy()
            data = self.get_data_pca(data)
            data = data[:60]
            survived = survived[:60]
            print(data.shape)
            # print(data[:5])

        elif(self.split == 'test'): 
            # print(df.isnull().sum())
            # print(df.loc[df['Fare'].isnull()].index)
            df['Fare'].fillna(0, inplace=True)
            # print(df.isnull().sum())

            data = df.to_numpy()
            data = self.get_data_pca(data)
            labels = np.zeros((len(data[:,0])))
            # data = self.get_data_pca(data)      
        
        # print(len(data[:, 0]))
        # print(data[190], labels[190])
        # print(labels.shape, data.shape)
        return data, labels
    
    def process_family(self, df):
        df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
        df['Single'] = df['FamilySize'].map(lambda s:1 if s == 1 else 0)
        df['SmallFamily'] = df['FamilySize'].map(lambda s:1 if 2 <= s <= 4 else 0)
        df['LargeFamily'] = df['FamilySize'].map(lambda s:1 if s >= 5 else 0)
        return df

    def process_embarked(self, df):
        df.Embarked.fillna('S', inplace=True)
        # df['Embarked'] = df['Embarked'].map(lambda c:'S' if c == "" else c)
        df_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
        df = pd.concat([df, df_dummies], axis=1)
        df.drop('Embarked', axis = 1, inplace=True)
        return df

    def process_cabin(self, df):
        # df.Cabin = df.Cabin.fillna('U')
        df['Cabin'].fillna('U', inplace=True)
        # df['Cabin'] = df['Cabin'].replace(np.nan , 'U')
        # df['Cabin'] = df['Cabin'].map(lambda c:'U' if c == "" else c)
        # print(df['Cabin'])
        df['Cabin'] = df['Cabin'].map(lambda c : c[0])
        df_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')
        df = pd.concat([df, df_dummies], axis=1)
        df.drop('Cabin', axis=1, inplace=True)
        return df

    def get_titles(self, df):
        Title_Dict = {
            'Capt':'Officer',
            'Col' : 'Officer',
            'Don' : 'Officer',
            'Dr' : 'Royalty', 
            'Jonkheer': 'Royalty',
            'Lady': 'Royalty',
            'Major' : "Officer",
            'Master' : 'Master',
            'Miss' : 'Miss',
            'Mlle' : 'Miss',
            'Mme' : 'Mrs',
            'Mr' : 'Mr',
            'Mrs' : 'Mrs',
            'Ms' : 'Mrs',
            'Rev' : 'Officer', 
            'Sir' : 'Royalty',
            'the Countess' : 'Royalty'
        }
        df['Title'] =  df['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
        df['Title'] = df.Title.map(Title_Dict)
        return df

    def fill_age(self, row, group_median_train):
        condition = (
            (group_median_train['Sex'] == row['Sex']) &
            (group_median_train['Title'] == row['Title']) &
            (group_median_train['Pclass'] == row['Pclass'])
        )
        if np.isnan(group_median_train[condition]['Age'].values[0]):
            print('true')
            condition = (
                (group_median_train['Sex'] == row['Sex']) &
                (group_median_train['Pclass'] == row['Pclass'])
            )
        return group_median_train[condition]['Age'].values[0]

    def get_age(self, df):
        group_train = df.groupby(['Sex', 'Pclass', 'Title'])
        group_median_train = group_train.median()
        # print(group_median_train.columns)
        group_median_train = group_median_train.reset_index()[['Sex', 'Pclass', 'Title','Age']]
        # print(group_median_train)
        df['Age'] = df.apply(lambda row: self.fill_age(row,group_median_train) if np.isnan(row['Age']) else row['Age'], axis = 1)
        return df

    def process_names(self, df):
        df.drop('Name', axis =1, inplace=True)

        title_dummies = pd.get_dummies(df['Title'], prefix='Title')
        df = pd.concat([df, title_dummies], axis=1)
        df.drop('Title', axis=1, inplace=True)
        return df


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
        
        ###  using first 23 components as they explain 100% of the dataset
        projection_matrix = (eigen_vectors.T[:][:22]).T
        # print(projection_matrix)

        #### Getting the product of original std data and projection matrix
        data_pca = data_std.dot(projection_matrix)
        # print(data_pca)
        return data_pca



##Testing the dataloader
if __name__ == '__main__':
    titanic_train_data = TitanicData('val', data_path='./dataset/train.csv')
    # print(titanic_train_data)
    data_loader = DataLoader(titanic_train_data, batch_size = 1, shuffle=False, num_workers = 0)
    for idx, item in enumerate(data_loader):
        data, label = item
        if(idx == 50):
            print(data.shape)
            print(label.shape)
            exit(0)
    # print(titanic_train_data)