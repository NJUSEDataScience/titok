# import plotly.plotly as plt
# import plotly.offline as pltoff
# from plotly.graph_objs import *
import glob
import os
from dataAnalysis import step_params
import numpy as np
from pyAudioAnalysis import audioTrainTest

def point_cal(l):
    mean1=np.sum(l)/len(l)
    m = list(map(lambda x: x * x, l))
    mean2=np.sum(m)/len(m)-mean1**2
    return mean1,mean2

l=[0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 6, 82, 1935, 8197, 8198, 8294, 8294, 8458, 8458, 9073, 12930, 15572, 15572, 18099, 23697, 23805, 24131, 24657, 24657, 25775, 27454, 28047, 28047, 32569, 38920, 45075, 45075, 49617, 51673, 66354, 68241, 70271, 70271, 70621, 76903, 80670, 80670, 83996, 83996, 110010, 110011, 110548, 114015, 115506, 120167, 150706, 150708, 172963, 186723, 190336, 190336, 199726, 221970, 222880, 222880, 267660, 282896, 282899, 284761, 284761, 331484, 345166, 345168, 379099, 413130, 420233, 420234, 440003, 448500, 448500, 547829, 547829, 612469, 612472, 639917, 719088, 1176491, 1176491, 1228956, 1228956, 1257544, 1379461, 1446799, 1446799, 1744101, 1771246, 2915208, 5924576, 16046526]

# print(point_cal(l))

def check(path,mode_path,mode_type,answer_name,check_name):
    path_v1 = os.path.join(path, "*")
    path_list = glob.glob(path_v1)
    f = open(answer_name + ".txt", 'w')
    for i in range(len(path_list)):
        class_id, probability, classes = audioTrainTest.file_classification(path_list[i], mode_path,mode_type)
        probability = probability.tolist()
        f.write(str(path_list[i]).split('/')[-1]+ "\n" + str(probability.index(max(probability)))+ "\n")
    f.close()
    f = open(answer_name + ".txt", 'r')
    answer = f.readlines()
    f.close()
    f = open(check_name + ".txt", 'r')
    check = f.readlines()
    total = len(answer)/2.0
    correct = 0
    for i in range(len(answer)):
        temindex = check.index(answer[i])
        if temindex!=-1 and answer[i+1]==check[temindex+2]:
            correct+=1
        i+=2
    print(correct/total)


def auto_test(path,mode_path,mode_type,answer_name,check_name):
    path_v1 = os.path.join(path, "*")
    path_list = glob.glob(path_v1)
    wav_file_list = [[] for i in range(len(path_list))]
    f = open(answer_name + ".txt", 'w')

    for num, i in enumerate(path_list):
        for string in step_params.container().get_music_type():
            x = os.path.join(i, string)
            y = glob.glob(x)
            wav_file_list[num].extend(y)
    for i in range(len(wav_file_list)):
        for j in range(len(wav_file_list[i])):
            class_id, probability, classes = audioTrainTest.file_classification(wav_file_list[i][j], mode_path, mode_type)
            probability = probability.tolist()
            f.write(str(wav_file_list[i][j]).split('/')[-1]+"\n"+str(probability.index(max(probability)))+"\n"+str(i)+"\n")
    f.close()


# auto_test("/Users/zhouhan/Downloads/音乐","/Users/zhouhan/Downloads/mode/svmmode/svm","svm","/Users/zhouhan/Downloads/2","/Users/zhouhan/Downloads/1")
check("/Users/zhouhan/Downloads/test","/Users/zhouhan/Downloads/mode/svmmode/svm","svm","/Users/zhouhan/Downloads/1","/Users/zhouhan/Downloads/2")
