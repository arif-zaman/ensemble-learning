# Arif Uz Zaman
# 1005031
# CSE,BUET

import math
import random

SamplesSet,trainData,testData,leftData = None,None,None,None


class ID3:
    def Entropy(self, dataSet, tCol):
        pCount = {}
        for row in dataSet:
            if row[tCol] in pCount:
                pCount[row[tCol]] += 1
            else:
                pCount[row[tCol]] = 1

        entropy = 0.0
        for key in pCount:
            temp1 = -float(pCount[key])/len(dataSet)
            temp2 = math.log(float(pCount[key])/len(dataSet), 2)
            entropy += temp1*temp2

        return entropy


    def Gain(self, dataSet, featureList, tCol):
        pCount = {}
        for row in dataSet:
            if row[featureList] in pCount:
                pCount[row[featureList]] += 1
            else:
                pCount[row[featureList]] = 1

        newEntropy = 0.0
        for key in pCount:
            prob = float(pCount[key]) / sum(pCount.values())
            newDataSet = [row for row in dataSet if row[featureList] == key]
            newEntropy += prob * self.Entropy(newDataSet, tCol)

        return (self.Entropy(dataSet, tCol) - newEntropy)


class DecisionTree:
    def Run(self, samples):
        featureList = [featureList for featureList in samples.pop(0)]
        tCol = featureList[-1]

        dataSet = []
        for row in samples:
            dataSet.append( dict( zip(featureList,[value for value in row]) ) )

        id3 = ID3()
        Tree = self.buildTree(dataSet, featureList, tCol, id3.Gain)
        key = list(Tree.keys())[0]
        value = Tree[key]
        Tree = {}
        Tree[key] = value

        return Tree,self.findClass(Tree,dataSet)


    def Unique(self, tColValues):
        uSet = set([])
        for item in tColValues:
            uSet.add(item)

        return list(uSet)


    def bestFeature(self, dataSet, featureList, tCol, id3Func):
        high,best = 0.0,None
        for feature in featureList:
            new = id3Func(dataSet, feature, tCol)
            if (new >= high and feature != tCol):
                high = new
                best = feature

        return best


    def bestValue(self, dataSet, tCol):
        matrix = [row[tCol] for row in dataSet]
        temp = 0
        for value in self.Unique(matrix):
            if matrix.count(value) > temp:
                Frequent = value
                temp = matrix.count(value)

        return Frequent


    def releventValues(self, dataSet, feature, value):
        dataSet = dataSet[:]
        relevent = []

        if not dataSet:
            return relevent
        else:
            row = dataSet.pop()
            if row[feature] == value:
                relevent.append(row)
                relevent.extend(self.releventValues(dataSet, feature, value))
                return relevent
            else:
                relevent.extend(self.releventValues(dataSet, feature, value))
                return relevent


    def buildTree(self, dataSet, featureList, tCol, id3Func):
        tMatrix = [row[tCol] for row in dataSet]

        if not dataSet or (len(featureList) - 1) <= 0:
            return self.bestValue(dataSet, tCol)

        elif tMatrix.count(tMatrix[0]) == len(tMatrix):
            return tMatrix[0]

        else:
            best = self.bestFeature(dataSet, featureList, tCol,id3Func)
            # Emplty Dictionary
            Tree = {best:{}}

            matrix = [row[best] for row in dataSet]
            for value in self.Unique(matrix):
                subtree = self.buildTree(self.releventValues(dataSet, best, value),[featureList for featureList in featureList if featureList != best],tCol,id3Func)
                Tree[best][value] = subtree

        return Tree


    def className(self, row, Tree):
        if type(Tree) == type("string"):
            return Tree
        else:
            featureList = Tree.keys()[0]
            t = Tree[featureList][row[featureList]]
            return self.className(row, t)


    def findClass(self, Tree, dataSet):
        return [self.className(row, Tree) for row in dataSet]


class Ensemble:
    Ws, nW, wL, bL = [],[],[],[]
    K = 0

    def __init__(self,K):
        self.K = K
        self.Run()


    def Run(self):
        object = self.adaBoost()
        return self.testEnsemble(object)

    def adaBoost(self):
        global trainData

        for i in xrange(len(trainData)):
            w = 1/float(len(trainData))
            self.Ws.append(w)

        dtree = DecisionTree()

        cost = sum(self.Ws)
        for i in xrange(self.K):
            self.nW, self.wL = [],[]
            wSum = 0

            for j in xrange(len(trainData)):
                try:
                    nw = self.Ws[j]/cost
                except:
                    pass

                self.nW.append(nw)
                wSum = wSum+nw
                self.wL.append(wSum)

            self.selectSamples()
            Tree,outputs = dtree.Run(trainData)
            error = 0.0
            length = min(len(outputs),len(trainData))
            for j in xrange(length):
                if outputs[j] != trainData[j][-1]:
                    error += self.nW[j]

            beta = error/(1-error)
            self.bL.append(beta)

            if (error>.5):
                break;

            cost = 0.0
            for j in xrange(length):
                if outputs[j] == trainData[j][-1]:
                    self.Ws[j] = self.Ws[j]*beta

                cost = cost + self.Ws[j]

        return dtree


    def selectSamples(self):
        global trainData,leftData
        w = 1/float(len(trainData))

        for i in xrange(len(trainData)):
            if (self.Ws[i]<w):
                if leftData:
                    trainData.pop(i)
                    trainData.insert(i,leftData.pop())
                    random.shuffle(leftData)


    def testEnsemble(self,object):
        count = 0
        Tree,outputs = object.Run(testData)
        length = min(len(outputs),len(testData))

        for i in xrange(len(testData)):
            alpha =  0.0
            for j in xrange(len(self.bL)):
                if outputs[i] == testData[i][-1]:
                    try:
                        alpha += math.log10(1/float(self.bL[j]))
                    except:
                        pass

            if alpha>0:
                count += 1

        Accuracy = float(count)/len(testData) * 100
        return Accuracy


def input():
    global SamplesSet
    with open("data.csv","r+") as F:
        SamplesSet = [ [value for value in row.replace("\n","").replace("\r", "").split(",")] for row in F.readlines() ]


def getAccuracy(K):
    global SamplesSet, trainData, testData, leftData

    random.shuffle(SamplesSet)
    num = int(.4*len(SamplesSet))
    trainData = SamplesSet[:num]
    leftData = SamplesSet[num:2*num]
    testData = SamplesSet[2*num:]

    ensemble = Ensemble(K)
    return ensemble.Run()


def main():
    random.seed(100)
    input()
    print("Now Calculating Output. Please Wait .... ")
    print("\t K \t\t Accuracy")

    for k in (5,10,20,30,):
        temp = []
        for y in range(k):
            temp.append(getAccuracy(k))
        print ("\t %2d \t %4.2f") % (k, max(temp))


if __name__ == '__main__':
    main()