# coding:utf8
import matplotlib.pyplot as plt
import numpy as np


def visNormal(myArr, deleteOutlier=True, lineColor="k", histColor="#e0620d", nameOfData="original data", partNum=5):
    '''
    数据的正态分布对比图：
    ==============================================================
    input:
    myArr: Inputted array输入数组
    deleteOutlier: Whether to delete outliers (default:True) 是否删除离群值（默认删除） (mean+/-3*sigma)
    lineColor: The color of standard normal distribution (default:"k") 标准正态分布（折线图）颜色（默认黑色）
    histColor: The color of true distribution of original data (default:"k") 原数据分布（柱状图）颜色（默认黑色）
    partNum: The number of the hists (default:5) 分段数（默认5）
    imgPath: The path used to save the visualization image (default:"normImg.jpg") 图像存储路径（默认 本地/normImg.jpg）
    ==============================================================
    return:plt
    '''
    if deleteOutlier == True:  # delete outliers 删除离群值
        tempMean = np.mean(myArr)
        tempStd = np.std(myArr)
        myArr = myArr[myArr != tempMean + 3 * tempStd]
        myArr = myArr[myArr != tempMean - 3 * tempStd]
    realMax = np.max(myArr)  # get the maximum 获取最大值
    realMin = np.min(myArr)  # get the minimum 获取最小值
    print("realMax", realMax)
    print("realMin",realMin)
    partItemScale = (realMax - realMin) / partNum  # get the unit distance 获取间隔单位距离
    print("part",partItemScale)
    myArrCoor = np.array([[(realMin + i * partItemScale + realMin + (i + 1) * partItemScale) / 2, myArr[
        np.logical_and.reduce([myArr >= realMin + i * partItemScale, myArr < realMin + (i + 1) * partItemScale])].size]
                          for i in range(partNum)])  # get the distribution of the coordinate data 获取原数据柱状分布
    pltBar = plt.bar(x=myArrCoor[:, 0], height=myArrCoor[:, 1], width=(realMax - realMin) / partNum, color=histColor,
                     edgecolor='#e0620d')  # plot 输出柱状图

    myMean = np.mean(myArr)  # get the mean of original data 获取原数据均值
    myStd = np.std(myArr)  # get the standard deviation of original data 获取原数据标准差
    normArr = np.random.normal(loc=myMean, scale=myStd,
                               size=myArr.size)  # get the data of normal distribution (ND)获取正态分布随机样本
    ndMin = np.min(normArr)  # get the minimum of ND 获取随机样本最小值
    ndMax = np.max(normArr)  # get the maxmum of ND 获取随机样本最大值
    ndStd = np.std(normArr)  # get the standard deviation of ND 获取随机样本标准差
    ndPartScale = (ndMax - ndMin) / partNum  # get the unit distance of ND 获取随机样本拐点间隔距离
    ndCoor = np.array([[(ndMin + i * ndPartScale + ndMin + (i + 1) * ndPartScale) / 2, normArr[
        np.logical_and.reduce([normArr >= ndMin + i * ndPartScale, normArr < ndMin + (i + 1) * ndPartScale])].size] for
                       i in range(partNum)])  # get the coordinate of ND 获取随机样本坐标
    pltLine, = plt.plot(ndCoor[:, 0], ndCoor[:, 1], c=lineColor, linestyle="--")  # plot 输出折线图

    plt.legend((pltBar, pltLine), (nameOfData, 'Normal Data'))  # set legend 设定标注


    return plt


if __name__ == "__main__":
    likes = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 6, 82, 1935, 8197, 8198, 8294, 8294, 8458, 8458, 9073, 12930, 15572, 15572,
             18099, 23697, 23805, 24131, 24657, 24657, 25775, 27454, 28047, 28047, 32569, 38920, 45075, 45075, 49617,
             51673, 66354, 68241, 70271, 70271, 70621, 76903, 80670, 80670, 83996, 83996, 110010, 110011, 110548,
             114015, 115506, 120167, 150706, 150708, 172963, 186723, 190336, 190336, 199726, 221970, 222880, 222880,
             267660, 282896, 282899, 284761, 284761, 331484, 345166, 345168, 379099, 413130, 420233, 420234, 440003,
             448500, 448500, 547829, 547829, 612469, 612472, 639917, 719088, 1176491, 1176491, 1228956, 1228956,
             1257544, 1379461, 1446799, 1446799, 1744101, 1771246, 2915208]
    shares = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 321, 381, 473, 610, 965, 1650, 1650, 1751, 1751, 1766, 1799,
              1799, 2014, 2014, 2149, 2337, 2453, 2507, 2507, 3300, 3300, 4589, 4589, 4610, 4966, 5618, 5857, 6208,
              6492, 6764, 7131, 9212, 9212, 10035, 10531, 10531, 11844, 13971, 13971, 14377, 14505, 15867, 17539, 17539,
              17731, 17731, 19644, 19703, 20420, 20527, 20685, 22521, 22521, 26503, 26503, 27131, 28134, 28134, 32127,
              32127, 32829, 36764, 36764, 39509, 41687, 52113, 52113, 53315, 53315, 55393, 58633, 58633, 69390, 69390,
              72730, 90642, 91855, 94909, 94909, 96181, 96181, 170415, 171702, 171702, 196496, 206130]
    comments = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 18, 61, 121, 121, 251, 272, 272, 297, 297, 347, 347, 384, 430, 433,
                433, 433, 546, 615, 626, 684, 726, 726, 880, 940, 963, 996, 1120, 1210, 1210, 1240, 1240, 1354, 1354,
                1429, 1429, 1493, 1517, 1640, 1728, 1728, 1837, 1897, 1897, 2026, 2128, 2252, 2252, 2329, 2329, 2412,
                3274, 3323, 3323, 3349, 3407, 4333, 4659, 5762, 5762, 6026, 6423, 6423, 6471, 6485, 7595, 10298, 10310,
                10310, 10657, 10657, 11540, 11540, 13216, 13216, 20806, 22019, 22019, 24221, 27484, 27484, 36514, 47475,
                51530, 59507, 59507, 89852, 180565, 662917]
    a=np.array(likes)
    myPlt = visNormal(a)
    plt.show()