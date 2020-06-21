# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : main.py
# @Item    : PyCharm
# @Time    : 2020-06-21 14:02
# @WebSite : https://www.gentlecp.com

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from model import RNNClassifier
from data import NameDataset
from config import Config
import util

def train(epoch):
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader,1):
        optimizer.zero_grad()
        inputs, seq_lengths, target = util.make_tensors(names, countries)
        output = model(inputs, seq_lengths)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % Config.print_batch_idx ==0:
            print('Epoch:{}, loss:{:.6f}'.format(epoch,total_loss/(i*len(inputs))))


def test(epoch):
    correct = 0
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader, 1):
            inputs, seq_lengths, target = util.make_tensors(names, countries)
            output = model(inputs, seq_lengths)
            pred=  output.max(dim=1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        print('Test: acc {:.4f}'.format(correct/len(test_loader.dataset)))

if __name__ == '__main__':
    train_dataset = NameDataset(train=True)
    test_dataset = NameDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    model = RNNClassifier(Config.n_chars, Config.hidden_size, train_dataset.get_country_num(),Config.n_layer)
    print(model)
    if Config.use_gpu:
        device = torch.device("cuda:0")
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = Config.lr)

    for epoch in range(1, Config.n_epochs+1):
        train(epoch)
        test(epoch)
