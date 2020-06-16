# @Author  : GentleCP
# @Email   : 574881148@qq.com
# @File    : word2vec_wiki_zh.py
# @Item    : PyCharm
# @Time    : 2020-06-16 23:19
# @WebSite : https://www.gentlecp.com


import re
import os
import time
from tqdm import tqdm

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import zhconv
import jieba

English_pattern = re.compile('[a-z]')  # 检测是否包含英文字母


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        time_cost = time.time() - start
        if 60 <= time_cost < 3600:
            # 不到1h
            print('time cost:{:4f} mins'.format(time_cost / 60))
        elif time_cost < 60:
            print('time cost:{:4f} s'.format(time_cost))
        else:
            print('time cost:{:4f} hours'.format(time_cost / 3600))

    return wrapper


def xml2txt(inp, outp, retransform=False):
    """
    处理原始文件-zhwiki-latest-pages-articles.xml.bz2成txt形式
    """
    if not os.path.exists(outp) or retransform:
        with open(outp, 'w', encoding='utf-8') as out:
            wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
            print('Transforming original file to txt ...(it may takes about 30-40 mins)')
            for texts in tqdm(wiki.get_texts()):  # 每个text是一个列表
                # 去除text中的英文单词
                no_english_text = []
                for text in texts:
                    if not bool(English_pattern.search(text)):
                        # 未检测到英文字母
                        no_english_text.append(text)
                line_text = ' '.join(no_english_text) + '\n'  # 拼接成一条字符串
                out.write(zhconv.convert(line_text, 'zh-cn'))  # 转换为简体输出

            print('Transform finished.')

    else:
        print('Already Transformed')


def cut_words(inp, outp, recut=False):
    """
    对txt文件进行分词，保存到分词结果文件
    """
    if not os.path.exists(outp) or recut:
        print('Cutting words...')
        out = open(outp, 'w', encoding='utf-8')
        with open(inp, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                cut_result = jieba.lcut(line.strip())
                line_result = ' '.join(cut_result) + '\n'
                out.write(line_result)
        out.close()
    else:
        print('Already cutted.')


@timer
def train_w2v_model(inp, outp1, outp2, retrain=False):
    """
    输入是预处理好的训练数据，按行输入到Word2Vec中训练，最后保存模型和向量

    """
    if not os.path.exists(outp1) or not os.path.exists(outp2) or retrain:
        print('Training w2v model...')
        model = Word2Vec(LineSentence(inp), size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())
        model.save(outp1)  # 用于加载
        model.wv.save_word2vec_format(outp2, binary=False)  # 用于查看
    else:
        print('w2v model alwarday trained.')


def test_model():
    print('loading model...')
    model = Word2Vec.load('./w2v_model/wiki_zh_w2v_model.bin')
    print('苹果的词向量:{}'.format(model['苹果']))
    print('与苹果最接近单词:{}'.format(model.wv.most_similar('苹果')))
    print('苹果与葡萄相似度:{}，苹果与计算机相似度:{}'.format(model.wv.similarity('苹果', '葡萄'), model.wv.similarity('苹果', '计算机')))
    print('国王-男人+女人的词向量结果:{}'.format(model.most_similar(positive=['女人', '国王'], negative=['男人'])))


def main():
    xml2txt('./data/zhwiki-latest-pages-articles.xml.bz2', './data/wiki_zh.txt')
    cut_words('./data/wiki_zh.txt', './data/wiki_zh_cut.txt')
    # train_w2v_model('./data/wiki_zh_cut.txt', './w2v_model/wiki_zh_w2v_model.bin', './w2v_model/wiki_zh_w2v_vector.txt')
    test_model()


if __name__ == '__main__':
    main()