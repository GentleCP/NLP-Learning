# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : util.py
# @Item    : PyCharm
# @Time    : 2020-06-21 15:03
# @WebSite : https://www.gentlecp.com

from config import Config
import torch

def create_tensor(tensor):
    """
    根据是否使用gpu，将数据添加到对应设备上
    :param tensor:
    :return:
    """
    if Config.use_gpu:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)

    return tensor


def make_tensors(names, countries):
    """
    将name和country转换成tensor表示
    :param names:
    :param countries:
    :return:
    """
    def name2list(name):
        """
        将名字转换成列表
        :param name: Acheo
        :return:[76,25,23,...], len of name
        """
        name_list = [ord(char) for char in name]
        return name_list, len(name_list)

    name_sequences_lengths = [name2list(name) for name in names]
    name_sequences = [s[0] for s in name_sequences_lengths]
    seq_lengths = torch.LongTensor([s[1] for s in name_sequences_lengths])

    # padding
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()  # 创建一个所有名字组成的最大tensor
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths)):
        seq_tensor[idx,:seq_len]  = torch.LongTensor(seq)  # 将对应数据粘过去

    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), create_tensor(seq_lengths),create_tensor(countries)

