import numpy as np
from utils.word_utils import *



class NaiveBayesBase(object):

    def __init__(self):
        pass


    def fit(self, trainMatrix, trainCategory):
        '''
        朴素贝叶斯分类器训练函数，求：p(Ci),基于词汇表的p(w|Ci)
        Args:
            trainMatrix : 训练矩阵，即向量化表示后的文档（词条集合）
            trainCategory : 文档中每个词条的列表标注
        Return:
            p0Vect : 属于0类别的概率向量(p(w1|C0),p(w2|C0),...,p(wn|C0))
            p1Vect : 属于1类别的概率向量(p(w1|C1),p(w2|C1),...,p(wn|C1))
            pAbusive : 属于1类别文档的概率
        '''
        numTrainDocs = len(trainMatrix)
        # 长度为词汇表长度
        numWords = len(trainMatrix[0])
        # p(ci)
        self.pAbusive = sum(trainCategory) / float(numTrainDocs)
        # 由于后期要计算p(w|Ci)=p(w1|Ci)*p(w2|Ci)*...*p(wn|Ci)，若wj未出现，则p(wj|Ci)=0,因此p(w|Ci)=0，这样显然是不对的
        # 故在初始化时，将所有词的出现数初始化为1，分母即出现词条总数初始化为2
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0
        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])
        # p(wi | c1)
        # 为了避免下溢出（当所有的p都很小时，再相乘会得到0.0，使用log则会避免得到0.0）
        self.p1Vect = np.log(p1Num / p1Denom)
        # p(wi | c2)
        self.p0Vect = np.log(p0Num / p0Denom)
        return self


    def predict(self, testX):
        '''
        朴素贝叶斯分类器
        Args:
            testX : 待分类的文档向量（已转换成array）
            p0Vect : p(w|C0)
            p1Vect : p(w|C1)
            pAbusive : p(C1)
        Return:
            1 : 为侮辱性文档 (基于当前文档的p(w|C1)*p(C1)=log(基于当前文档的p(w|C1))+log(p(C1)))
            0 : 非侮辱性文档 (基于当前文档的p(w|C0)*p(C0)=log(基于当前文档的p(w|C0))+log(p(C0)))
        '''

        p1 = np.sum(testX * self.p1Vect) + np.log(self.pAbusive)
        p0 = np.sum(testX * self.p0Vect) + np.log(1 - self.pAbusive)
        if p1 > p0:
            return 1
        else:
            return 0

def loadDataSet():
    '''数据加载函数。这里是一个小例子'''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论，代表上面6个样本的类别
    return postingList, classVec


def checkNB():
    '''测试'''
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postDoc in listPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postDoc))

    nb = NaiveBayesBase()
    nb.fit(np.array(trainMat), np.array(listClasses))

    testEntry1 = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWord2Vec(myVocabList, testEntry1))
    print(testEntry1, 'classified as:', nb.predict(thisDoc))

    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = np.array(setOfWord2Vec(myVocabList, testEntry2))
    print(testEntry2, 'classified as:', nb.predict(thisDoc2))


if __name__ == "__main__":
    checkNB()