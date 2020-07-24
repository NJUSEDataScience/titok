from __future__ import print_function
import numpy as np
import pickle as cPickle
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioTrainTest as aT
import step_params
import os

def freature_extraction(paths, mid_window, mid_step, short_window, short_step,useage):
    """

    :param paths: need to be a list of string, like ['F:\\tempFile\pythonWav\people','F:\\tempFile\pythonWav\\nopeople']
                 as you can see, each object contains a different kind of music type
    The following four: units: in seconds
    suppose the mid_window is five,short_window is 1,each mid-term window will consider ceil(5/1)=5 as a whole to calculate
    :param mid_window: the whole length of mid-term window you would like to deal with
    :param mid_step: moving ${mid_step} once
    :param short_window:
    :param short_step:
    :param useage: the feature you would kike to use
    :return: a numpy array of sum(mi*k); mi means the number of files contained in the ith directory;k is the number of
    feature. (maximum=2*2*34(28 different feature nest 34 cols, multiplied by 2(origin and its delta)*2(std and mean) ))

    """
    [features, class_names, feature_names,_]= multiple_directory_feature_extraction(paths, mid_window, mid_step,
                                                 short_window, short_step,False)
    features=np.array(features)
    tep_features=np.empty([features.shape[0],features.shape[1],1])
    for need in useage:
        for i,feature_name in enumerate(feature_names):
            if need==feature_name or feature_name.find(need)!=-1:
                tep_features=np.concatenate((tep_features,features[:,:,2:3]),2)

    tep_features=np.delete(tep_features,0,2)
    return tep_features

def multiple_directory_feature_extraction(path_list, mid_window, mid_step,
                                          short_window, short_step,
                                          compute_beat=False):
    """
   this version is used to get the whole info of features, class_names, feature_names,file_names
    """
    # feature extraction for each class:
    features = []
    class_names = []
    file_names = []
    for i, d in enumerate(path_list):
        f, fn, feature_names = \
            aF.directory_feature_extraction(d, mid_window, mid_step,
                                         short_window, short_step,
                                         compute_beat=compute_beat)
        if f.shape[0] > 0:
            # if at least one audio file has been found in the provided folder:
            features.append(f)
            file_names.append(fn)
            print(feature_names)
            if d[-1] == os.sep:
                class_names.append(d.split(os.sep)[-2])
            else:
                class_names.append(d.split(os.sep)[-1])
    return features, class_names, feature_names,file_names


def train_svm(paths, mid_window, mid_step, short_window,
                               short_step, model_name,
                               compute_beat=False, train_percentage=0.90):


    # STEP A: Feature Extraction:
    [features, class_names, _]= aF.multiple_directory_feature_extraction(paths, mid_window, mid_step,
                                                 short_window, short_step,
                                                 compute_beat=compute_beat)

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    write_modearff_file(model_name, features, class_names, feature_names)
    # STEP B: classifier Evaluation and Parameter Selection:
    # get optimal classifeir parameter:
    temp_features = []
    for feat in features:
        temp = []
        for i in range(feat.shape[0]):
            temp_fv = feat[i, :]
            if (not np.isnan(temp_fv).any()) and (not np.isinf(temp_fv).any()):
                temp.append(temp_fv.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        temp_features.append(np.array(temp))
    features = temp_features

    best_param = aT.evaluate_classifier(features, class_names, 100,"svm",step_params.container.get_svm(), 0, train_percentage)

    print("Selected params: {0:.5f}".format(best_param))

    features_norm, mean, std = normalize_features(features)
    mean = mean.tolist()
    std = std.tolist()

    # STEP C: Save the classifier to file
    classifier = aT.train_svm(features_norm, best_param)
    write_mode_file(model_name,classifier, mean, std, class_names, mid_window, mid_step,short_window, short_step, compute_beat)

def train_knn(paths, mid_window, mid_step, short_window,
                               short_step, model_name,
                               compute_beat=False, train_percentage=0.90):


    # STEP A: Feature Extraction:
    [features, class_names, _]= aF.multiple_directory_feature_extraction(paths, mid_window, mid_step,
                                                 short_window, short_step,
                                                 compute_beat=compute_beat)

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    write_modearff_file(model_name, features, class_names, feature_names)
    # STEP B: classifier Evaluation and Parameter Selection:
    # get optimal classifeir parameter:
    temp_features = []
    for feat in features:
        temp = []
        for i in range(feat.shape[0]):
            temp_fv = feat[i, :]
            if (not np.isnan(temp_fv).any()) and (not np.isinf(temp_fv).any()):
                temp.append(temp_fv.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        temp_features.append(np.array(temp))
    features = temp_features

    best_param = aT.evaluate_classifier(features, class_names, 100,"knn",step_params.container.get_knn(), 0, train_percentage)

    print("Selected params: {0:.5f}".format(best_param))

    features_norm, mean, std = normalize_features(features)
    mean = mean.tolist()
    std = std.tolist()

    # STEP C: Save the classifier to file
    classifier = aT.train_knn(features_norm, best_param)
    write_mode_file(model_name,classifier, mean, std, class_names, mid_window, mid_step,short_window, short_step, compute_beat)


def write_mode_file(model_name,classifier, *parameters):
    with open(model_name, 'wb') as fid:
        cPickle.dump(classifier, fid)
    save_path = model_name + "MEANS"
    fid.close()
    with open(save_path, 'wb') as file_handle:
        for param in parameters:
            cPickle.dump(param, file_handle, protocol=cPickle.HIGHEST_PROTOCOL)
    file_handle.close()

def write_modearff_file(model_name, features, classNames, feature_names):
    f = open(model_name + ".arff", 'w')
    f.write('@RELATION ' + model_name + '\n')
    for fn in feature_names:
        f.write('@ATTRIBUTE ' + fn + ' NUMERIC\n')
    f.write('@ATTRIBUTE class {')
    for c in range(len(classNames)-1):
        f.write(classNames[c] + ',')
    f.write(classNames[-1] + '}\n\n')
    f.write('@DATA\n')
    for c, fe in enumerate(features):
        for i in range(fe.shape[0]):
            for j in range(fe.shape[1]):
                f.write("{0:f},".format(fe[i, j]))
            f.write(classNames[c]+"\n")
    f.close()


def normalize_features(features):
    """
    this function is used to minimize the amount of calculation
    """
    temp_feats = np.array([])
    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                temp_feats = f
            else:
                temp_feats = np.vstack((temp_feats, f))
            count += 1

    #归一化并且加上一个最小值防止原值为0而产生运算错误
    mean = np.mean(temp_feats, axis=0) + 1e-14
    std = np.std(temp_feats, axis=0) + 1e-14

    features_norm = []
    for f in features:
        ft = f.copy()
        for n_samples in range(f.shape[0]):
            ft[n_samples, :] = (ft[n_samples, :] - mean) / std
        features_norm.append(ft)
    return features_norm, mean, std