# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : data.py
# @Item    : PyCharm
# @Time    : 2020-06-21 14:03
# @WebSite : https://www.gentlecp.com

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from config import Config

class NameDataset(Dataset):
    """
    Name 数据集，一次返回name和国家标签
    """
    def __init__(self, train=False):
        self.data_path = './data/names_train.csv' if train else './data/names_test.csv'
        self.names, self.countries = self._read_data()
        self.country2id = self._get_country2id()

    def _read_data(self):
        df_data = pd.read_csv(self.data_path)
        return df_data['Name'].tolist(),df_data['Country'].tolist()

    def _get_country2id(self):
        countries = set(self.countries)
        country2id = {}
        for id, country in enumerate(countries):
            country2id[country] = id
        return country2id

    def get_country_num(self):
        return len(self.country2id.keys())

    def __getitem__(self, index):
        return self.names[index],self.country2id[self.countries[index]]

    def __len__(self):
        return len(self.names)




if __name__ == '__main__':
    train_dataset = NameDataset(train=True)
    test_dataset = NameDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    print(train_dataset.get_country_num())
    for name,country in train_loader:
        print(name,country)
        break

