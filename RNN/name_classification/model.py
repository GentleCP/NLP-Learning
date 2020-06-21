# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : model.py
# @Item    : PyCharm
# @Time    : 2020-06-21 14:44
# @WebSite : https://www.gentlecp.com
from torch import nn

import torch
import util

class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, hidden_size, output_size, n_layers= 1, bidirectional = True):
        """
        :param vocab_size: 输入数据的词数量
        :param hidden_size: 隐藏层表示维度
        :param output_size: 输出维度，即分类的数量
        :param n_layers:
        :param bidirectional:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 将词转换成hidden_size维度的向量表示
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)  # 输入hidden_size
        self.fc = nn.Linear(hidden_size* self.n_directions, output_size)  #  输入是GRU的输出，如果是双向则维度*2


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions,batch_size, self.hidden_size)
        return util.create_tensor(hidden)


    def forward(self, input, seq_lens):
        # input: batch * seq_len
        input = input.t()  # b*s -> s*b
        batch_size = input.size(1)  # 获取batch_size 用于创建初始隐藏层
        hidden = self.init_hidden(batch_size)  #

        embedding = self.embedding(input)  # 将input转换成词向量表示，seq_len * batch * hidden_size

        gru_input = nn.utils.rnn.pack_padded_sequence(embedding, seq_lens)  # 将embedding结果中pad添加的0不加入运算
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1],hidden[-2]], dim=1)
        else:
            hidden_cat =hidden[-1]
        fc_out = self.fc(hidden_cat)
        return fc_out