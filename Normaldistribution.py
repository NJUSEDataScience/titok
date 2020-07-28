# coding:utf8
import matplotlib.pyplot as plt
import numpy as np


def visNormal(myArr, deleteOutlier=True, lineColor="k", histColor="#696969", nameOfData="original data", partNum=5):
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
    partItemScale = (realMax - realMin) / partNum  # get the unit distance 获取间隔单位距离
    myArrCoor = np.array([[(realMin + i * partItemScale + realMin + (i + 1) * partItemScale) / 2, myArr[
        np.logical_and.reduce([myArr >= realMin + i * partItemScale, myArr < realMin + (i + 1) * partItemScale])].size]
                          for i in range(partNum)])  # get the distribution of the coordinate data 获取原数据柱状分布
    pltBar = plt.bar(x=myArrCoor[:, 0], height=myArrCoor[:, 1], width=(realMax - realMin) / partNum, color=histColor,
                     edgecolor='w')  # plot 输出柱状图

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
    a = np.random.normal(loc=3000, scale=50, size=5000)
    myPlt = visNormal(a)
    plt.show()