import pandas as pd
import string
import copy

# words = corpus()
# Reading Text Data
data = pd.read_csv("NaiveBayesDataset.csv");

# 5-fold cross validation
# Spliting data into 5 chunks of 200 each

x1 = data.iloc[0:200,:]
x2 = data.iloc[200:400,:]
x3 = data.iloc[400:600,:]
x4 = data.iloc[600:800,:]
x5 = data.iloc[800:1000,:]

xTrainSize = 800
xTestSize = 200
words = []

def fiveFoldCV():
    # Model 1
    xTrain = pd.concat([x1,x2,x3,x4],ignore_index=True).values
    xTest  = x5.values
    trainTest(xTrain,xTest,1)

    # Model 2
    xTrain = pd.concat([x1,x2,x3,x5],ignore_index=True).values
    xTest  = x4.values
    trainTest(xTrain,xTest,2)

    # Model 3
    xTrain = pd.concat([x1,x2,x4,x5],ignore_index=True).values
    xTest  = x3.values
    trainTest(xTrain,xTest,3)

    # Model 4
    xTrain = pd.concat([x1,x3,x4,x5],ignore_index=True).values
    xTest  = x2.values
    trainTest(xTrain,xTest,4)

    # Model 5
    xTrain = pd.concat([x2,x3,x4,x5],ignore_index=True).values
    xTest  = x1.values
    trainTest(xTrain,xTest,5)



def trainTest(xTrain,xTest,i):

    vocab = createVocab(xTrain)

    posDict,posCommentCount,negDict,negCommentCount = DictCount(xTrain,vocab)

    posWords = 0
    negWords = 0

    for key in posDict:
        posWords += posDict[key]

    for key in negDict:
        negWords += negDict[key]

    for key in posDict:
        posDict[key] = (posDict[key]+1)/(posWords+2)

    for key in negDict:
        negDict[key] = (negDict[key]+1)/(negWords+2)

    pPos = posCommentCount/xTrainSize
    pNeg = negCommentCount/xTrainSize   

    prediction,fScore,accuracy = test(xTest,pPos,pNeg,posDict,negDict)

    print("Model ",i)
    print("F Score: ",fScore)
    print("Accuracy: ",accuracy,"\n")   

def preprocess(reviews):

    # spliting into words
    tokens = reviews.split()
    # converting into tokens
    tokens = [w.lower() for w in tokens]
    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    
    stripped = [w.translate(table) for w in tokens]
    # Removing all non alphabatic tokens
    words = [word for word in stripped if word.isalpha()]

    return words


def DictCount(xTrain,vocab):
    posDict = copy.deepcopy(vocab)
    negDict = copy.deepcopy(vocab)
    pos = 0
    neg = 0
    for i in range(xTrainSize):
        if(xTrain[i,1]==1):
            pos = pos + 1
            tokens = preprocess(xTrain[i,0])

            for j in range(len(tokens)):
                try: 
                    posDict[tokens[j]] += 1
                except KeyError:
                    continue

        elif(xTrain[i,1]==0):
            neg = neg + 1
            tokens = preprocess(xTrain[i,0])

            for j in range(len(tokens)):
            
                try:
                    negDict[tokens[j]] += 1
                except KeyError:
                    continue
                    

    return posDict,pos,negDict,neg


# precision = true positive/ total predicted positive = true pos./true pos. + false pos.
# recall = true positive/Actual positive = true pos./true pos + false neg.
def test(xTest,pPos,pNeg,posDict,negDict):
    truePos = 0
    falsePos = 0
    trueNeg  = 0
    falseNeg = 0

    prediction = []
    for i in range(xTestSize):
        y0 = pNeg
        y1 = pPos
        tokens = preprocess(xTest[i,0])
        for j in range(len(tokens)):
            try:
                y1 *= posDict[tokens[j]]
                y0 *= negDict[tokens[j]]
            except KeyError:
                continue

        if(y1>y0):
            prediction.append(1)
            if(xTest[i,1]==1):
                truePos += 1
            elif(xTest[i,1]==0):
                falsePos +=1
    
        elif(y0>y1):
            prediction.append(0)
            if(xTest[i,1]==1):
                falseNeg +=1
            elif(xTest[i,1]==0):
                trueNeg +=1

    precision = truePos/(truePos+falsePos)
    recall    = truePos/(truePos+falseNeg) 
    fScore = 2*(precision*recall)/(precision+recall)  
    accuracy  = (truePos+trueNeg)/xTestSize
        
    return prediction,fScore,accuracy

# Creating vocab
def createVocab(xTrain):
    reviews = ""
    for i in range(xTrainSize):
        reviews += xTrain[i,0]
        reviews += " "

    words = preprocess(reviews)
    words = list(set(words))
    vocab = {}
    for i in range(len(words)):
        vocab[words[i]]=0
    return vocab

fiveFoldCV()
