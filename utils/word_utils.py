def createVocabList(dataSet):
    '''
    创建所有文档中出现的不重复词汇列表
    Args:
        dataSet: 所有文档
    Return:
        包含所有文档的不重复词列表，即词汇表
    '''
    vocabSet = set([])
    # 创建两个集合的并集
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 词袋模型(bag-of-words model):词在文档中出现的次数
def bagOfWords2Vec(vocabList, inputSet):
    '''
    依据词汇表，将输入文本转化成词袋模型词向量
    Args:
        vocabList: 词汇表
        inputSet: 当前输入文档
    Return:
        returnVec: 转换成词向量的文档
    例子：
        vocabList = ['I', 'love', 'python', 'and', 'machine', 'learning']
        inputset = ['python', 'machine', 'learning', 'python', 'machine']
        returnVec = [0, 0, 2, 0, 2, 1]
        长度与词汇表一样长，出现了的位置为1，未出现为0，如果词汇表中无该单词则print
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
        return returnVec


# 词集模型(set-of-words model):词在文档中是否存在，存在为1，不存在为0
def setOfWord2Vec(vocabList, inputSet):
    '''
    依据词汇表，将输入文本转化成词集模型词向量
    Args:
        vocabList: 词汇表
        inputSet: 当前输入文档
    Return:
        returnVec: 转换成词向量的文档
    例子：
        vocabList = ['I', 'love', 'python', 'and', 'machine', 'learning']
        inputset = ['python', 'machine', 'learning']
        returnVec = [0, 0, 1, 0, 1, 1]
        长度与词汇表一样长，出现了的位置为1，未出现为0，如果词汇表中无该单词则print
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec


