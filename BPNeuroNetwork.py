import operator
import numpy as np
import math
import random
import os
import types


#随机抽取70%作为测试数据集
def randomDivision(titanic):
    titanic1 = titanic
    for i in range(len(titanic)):
        if titanic[i][3] == 1:
            titanic1[i][3] = 1
        elif titanic[i][3] == -1:
            titanic1[i][3] = 0
            
    y = list(range(0,2201))
    split = random.sample(y,1541)
    test = list()
    training = list()
    for testNumber in range(0,2201):
        if(testNumber in split):
            continue
        else:
            test.append(titanic1[testNumber])

    for trainingNumber in split:
        training.append(titanic1[trainingNumber])

    test = np.array(test)
    training = np.array(training)

    return test,training


#定义sigmoid函数
def sigmoid(x):
    a = float(1)/(1+math.exp(-x))

    return a

def sigmodDerivate(x):
    a = sigmoid(x)*(1-sigmoid(x))

    return a

#权值初始化
def BPNetworkInitialize(training):
    trainingNum = len(training[0])-1
    hideNum = 4
    outNum = 1
    
    inputWeight = np.zeros([trainingNum,hideNum],dtype = 'float')
    for i in range(trainingNum):
        for j in range(hideNum):
            weight = random.uniform(-1,1)
            inputWeight[i][j] = weight

    hideWeight = np.zeros([hideNum,outNum],dtype = 'float')
    for i in range(hideNum):
        for j in range(outNum):
            weight = random.uniform(-1,1)
            hideWeight[i][j] = weight

    return inputWeight,hideWeight

#训练BP神经网络
def BPNetwork(training,inputWeight,hideWeight):
    trainingNum = len(training[0])-1
    hideNum = 4
    outNum = 1
    hideOut = np.zeros([len(training),hideNum],dtype = 'float')
    out = np.zeros([len(training),outNum],dtype = 'float')
    for i in range(len(training)):
        for j in range(hideNum):
            hideScore = training[i][0] * inputWeight[0][j] + training[i][1] * inputWeight[1][j] + training[i][2] * inputWeight[2][j]
            hideOut[i][j] = sigmoid(hideScore)

    for m in range(len(training)):
        for n in range(outNum):
            outScore = hideOut[m][0] * hideWeight[0][n] + hideOut[m][1] * hideWeight[1][n] + hideOut[m][2] * hideWeight[2][n] + hideOut[m][3] * hideWeight[3][n]
            out[m][n] = sigmoid(outScore)

    error = 0
    for i in range(len(training)):
        error += (out[i][0] - training[i][3])**2

    return error,hideOut,out

#重新修正权值
def BPWeightRecaculate(hideOut,out,inputWeight,hideWeight,training):
    studyRate = 0.05
    
    errorOut = np.zeros([len(out),1],dtype = 'float')
    totalErrorOut = 0
    totalOut = 0
    for i in range(len(out)):
        errorOut[i][0] = out[i][0] * (1 - out[i][0]) * (training[i][3] - out[i][0])
        totalErrorOut += errorOut[i][0]
        totalOut += out[i][0]

    for m in range(len(hideWeight)):
        hideWeight[m][0] += studyRate * (totalErrorOut / len(out)) * (totalOut / len(out))

    errorHide = np.zeros([len(hideOut),len(hideOut[0])],dtype = 'float')
    totalErrorHide = np.zeros([1,len(hideOut[0])],dtype = 'float')
    totalHide = np.zeros([1,len(hideOut[0])],dtype = 'float')
    for j in range(len(hideOut)):
        for k in range(len(hideOut[0])):
            errorHide[j][k] = hideOut[j][k] * (1 - hideOut[j][k]) * errorOut[j][0] * hideWeight[k][0]
            totalErrorHide[0][k] += errorHide[j][k]
            totalHide[0][k] += hideOut[i][k]

    for p in range(len(inputWeight)):
        for q in range(len(inputWeight[0])):
            inputWeight[p][q] += studyRate * (totalErrorHide[0][q] / len(hideOut)) * (totalHide[0][q] / len(hideOut))

    return inputWeight,hideWeight


#测试正确率
def AccuracyRate(test,inputWeight,hideWeight):
    prediction = np.zeros([len(test),1],dtype = 'float')
    hideOut = np.zeros([len(test),4],dtype = 'float')
    for i in range(len(test)):
        for j in range(len(inputWeight)):
            hideScore = test[i][0] * inputWeight[0][j] + test[i][1] * inputWeight[1][j] + test[i][2] * inputWeight[2][j]
            hideOut[i][j] = sigmoid(hideScore)
        
    for m in range(len(test)):
        outScore = hideOut[m][0] * hideWeight[0][0] + hideOut[m][1] * hideWeight[1][0] + hideOut[m][2] * hideWeight[2][0] + hideOut[m][3] * hideWeight[3][0]
        out = sigmoid(outScore)
        if out >= 0.5:
            prediction[m][0] = 1
        else:
            prediction[m][0] = 0

    right = 0
    wrong = 0
    for k in range(len(test)):
        if test[k][3] == prediction[k][0]:
            right += 1
        else:
            wrong += 1
            
    accurateRate = float(right)/(right+wrong)
    
    return accurateRate
    
    #实现随机梯度下降
def BPtraining(training):
    x = list(range(0,1541))
    split = random.sample(x,100)
    train = list()
    for i in split:
        train.append(training[i])
    train = np.array(train)

    return train



def mainBP():
    check1=1
    while (check1==1):
        file_address=input("please input the file:")
        if os.path.exists(file_address):
            check1=0
        else:
            print("cannot open the file: %s\n" % file_address)
    titanic = list()
    with open(file_address) as sourceFile:
        for line in sourceFile.readlines():
            if(line[0]=='@'):
                continue
            else:
                lineArr = line.strip().split(',')
                data = [float(item) for item in lineArr]
                titanic.append(data)
                sourceFile.close()
    
    titanic = np.array(titanic)

    test,training = randomDivision(titanic)

    inputWeight,hideWeight = BPNetworkInitialize(training)

    print("initial input weight:\n")
    print(inputWeight)
    print("initial hide weight:\n")
    print(hideWeight)

    error,hideOut,out = BPNetwork(training,inputWeight,hideWeight)

    testError = 0

    while testError - error > 0.001 or testError - error < -0.001:
        testError = error
        
        inputWeight,hideWeight = BPWeightRecaculate(hideOut,out,inputWeight,hideWeight,training)

        error,hideOut,out = BPNetwork(training,inputWeight,hideWeight)

    print("recaculate input weight\n")
    print(inputWeight)
    print("recaculate hide weight:\n")
    print(hideWeight)

    accurateRate = AccuracyRate(test,inputWeight,hideWeight)
    print(accurateRate)



def mainBPStochastic():
    check1=1
    while (check1==1):
        file_address=input("please input the file:")
        if os.path.exists(file_address):
            check1=0
        else:
            print("cannot open the file: %s\n" % file_address)
    titanic = list()
    with open(file_address) as sourceFile:
        for line in sourceFile.readlines():
            if(line[0]=='@'):
                continue
            else:
                lineArr = line.strip().split(',')
                data = [float(item) for item in lineArr]
                titanic.append(data)
                sourceFile.close()
    
    titanic = np.array(titanic)

    test,training = randomDivision(titanic)

    inputWeight,hideWeight = BPNetworkInitialize(training)

    print("initial input weight:\n")
    print(inputWeight)
    print("initial hide weight:\n")
    print(hideWeight)

    error,hideOut,out = BPNetwork(training,inputWeight,hideWeight)

    train = BPtraining(training)

    for i in range(len(train)):#训练并且得到结果
        
        inputWeight,hideWeight = BPWeightRecaculate(hideOut,out,inputWeight,hideWeight,training)

        error,hideOut,out = BPNetwork(training,inputWeight,hideWeight)
        

    print("recaculate input weight:\n")
    print(inputWeight)
    print("recaculate hide weight:\n")
    print(hideWeight)

    accurateRate = AccuracyRate(test,inputWeight,hideWeight)
    print(accurateRate)


def main():
    choice=""
    choice=input("BPStochastic or BP?")
    if choice == "BPStochastic" or "BP Stochastic":
        mainBPStochastic()

    elif choice == "BP":
        mainBP()

    else:
        choice == input("please input 'BPStochastic' or 'BP'")

    return choice    
    


if __name__ == "__main__":
    main()


