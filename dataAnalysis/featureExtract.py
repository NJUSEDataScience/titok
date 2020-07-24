
import os
from pyAudioAnalysis import audioTrainTest
# [Fs, x] = audioBasicIO.read_audio_file("F:\\tempFile\pythonWav\sample1.wav")
# x=audioBasicIO.stereo_to_mono(x)
# F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.50*Fs, 0.25*Fs)
#
# plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0])
# plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()

# [mid_features, short_features, mid_feature_names]=MidTermFeatures.directory_feature_extraction("F:\\tempFile\pythonWav",10,10,1,1,False)

# [features, classNames,_] =\
#            MidTermFeatures.multiple_directory_feature_extraction(['F:\\tempFile\pythonWav\people','F:\\tempFile\pythonWav\\nopeople'], 10,10, 0.2, 0.2)
from pyAudioAnalysis.MidTermFeatures import directory_feature_extraction
import training
# answer=training.freature_extraction(['F:\\tempFile\pythonWav\people','F:\\tempFile\pythonWav\\nopeople'], 10,10, 0.2, 0.2,['zcr_mean', 'energy_mean'])
# audioTrainTest.extract_features_and_train(['F:\\tempFile\pythonWav\people','F:\\tempFile\pythonWav\\nopeople'], 10,10, 0.2, 0.2,"svm","F:\\tempFile\mode_test\\v2\mode")

class_id, probability, classes=audioTrainTest.file_classification("F:\\tempFile\pythonWav\\nopeople\\05-风起天阑.wav","F:\\tempFile\mode_test\\v1\mode","svm")
print(class_id, probability, classes)
# a=1

import visual
visual.generateFeatures_Single("F:\\tempFile\pythonWav\\nopeople\\05-风起天阑.wav",["zcr"])
# visual.generate_CompareGraph()
