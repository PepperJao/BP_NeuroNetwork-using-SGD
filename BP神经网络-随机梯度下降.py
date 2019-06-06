import operator
import numpy as np
import math
import random
import os
import types
#从BP梯度下降代码BP.py中调用
from BP import randomDivision, sigmoid, sigmoidDerivate, BPNetworkInitialize, BPNetwork, AccuracyRate


#实现随机梯度下降
def BPtraining(training):
    x = list(range(0,1541))
    split = random.sample(x,100)
    train = list()
    for i in split:
        train.append(training[i])
    train = np.array(train)

    return train



def main():
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

    for i in range(len(train)):#随机训练并且得到结果
        
        inputWeight,hideWeight = BPWeightRecaculate(hideOut,out,inputWeight,hideWeight,training)

        error,hideOut,out = BPNetwork(training,inputWeight,hideWeight)
        

    print("recaculate input weight:\n")
    print(inputWeight)
    print("recaculate hide weight:\n")
    print(hideWeight)

    accurateRate = Test(test,inputWeight,hideWeight)
    print(accurateRate)


if __name__ == "__main__":
    main()
