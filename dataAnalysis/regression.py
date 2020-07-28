import glob
import os
from pyAudioAnalysis import audioTrainTest
from dataAnalysis import step_params


def auto_test_regression(path,mode_path,mode_type,answer_name):
    path_v1 = os.path.join(path, "*")
    path_list = glob.glob(path_v1)
    path_list.pop(0)
    wav_file_list = [[] for i in range(len(path_list))]
    f = open(answer_name + ".txt", 'w')

    # for num, i in enumerate(path_list):
    #     for string in step_params.container().get_music_type():
    #         x = os.path.join(i, string)
    #         y = glob.glob(x)
    #         wav_file_list[num].extend(y)
    audioTrainTest.feature_extraction_train_regression(path_list[0], 10, 10, 0.2, 0,2,mode_type,"svm1")
    f.close()

auto_test_regression("/Users/zhouhan/Downloads/音乐","/Users/zhouhan/Downloads/mode/svmmode/svm","svm","/Users/zhouhan/Downloads/1")