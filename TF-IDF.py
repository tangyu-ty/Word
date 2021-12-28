#!/usr/bin/python
# coding=utf-8
# 采用TF-IDF方法提取文本关键词
# http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
import sys, codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

"""
       TF-IDF权重：
           1、CountVectorizer 构建词频矩阵
           2、TfidfTransformer 构建tfidf权值计算
           3、文本的关键字
           4、对应的tfidf矩阵
"""


# 数据预处理操作：分词，去停用词，词性筛选
# test：内容
# stopkey：停止key
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    seg = jieba.posseg.cut(text)  # 分词，顺便取出来词性
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    return l


# tf-idf获取文本top10关键词
# data：数据源
# stopley：停止词
# topK：前几个
def getKeywords_tfidf(data, stopkey, topK):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    corpus = []  # 将所有文档输出到一个list中，一行就是一个文档
    for index in range(len(idList)):
        text = f'{titleList[index]}。{abstractList[index]}'  # 拼接标题和摘要
        text = dataPrepos(text, stopkey)  # 文本预处理
        text = " ".join(text)  # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()

    # CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。
    # CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数。

    X = vectorizer.fit_transform(corpus)  # 词频矩阵,X[i][j]:表示j词在第i个文本中的词频

    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)  # (X[i][j],tf-idf)

    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    print(word)  # 按照value顺序来输出key单词。
    print(vectorizer.vocabulary_)  # 按照文档顺序，也就是key来输出单词
    # 其中，value的数字是按照字典序a-z来排序的。
    # 4、获取tf-idf矩阵，X[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()  # 将原来的(i,j) ,tf_idf 改为[tf_idf]矩阵，ij变成矩阵下标，tf_idf变为值

    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range( len(weight)):
        print(u"-------这里输出第", i + 1, u"篇文本的词语tf-idf------")
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word, df_weight = [], []  # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            print(word[j], weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1)  # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight", ascending=False)  # 按照权重值降序排列
        keyword = np.array(word_weight['word'])  # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0, topK)]  # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        print("word_split:")
        print(word_split)
        keys.append(word_split)
        print("keys")
        print(keys)

    result = pd.DataFrame({"id": ids, "title": titles, "key": keys}, columns=['id', 'title', 'key'])
    return result


def main():
    # 读取数据集
    dataFile = 'data/sample_data.csv'
    data = pd.read_csv(dataFile)
    # 停用词表
    stopkey = [w.strip() for w in codecs.open('data/stopWord.txt', 'r', 'utf-8').readlines()]  # 返回所有行，
    # str.strip([str]),移除字符串头尾指定的字符序列[str]
    # tf-idf关键词抽取
    result = getKeywords_tfidf(data, stopkey, 10)
    result.to_csv("result/keys_TFIDF.csv", index=False)


if __name__ == '__main__':
    main()
