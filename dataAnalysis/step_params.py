import numpy as np
class container:
    """
    this class is designed according to VUE_Store style
    we refactor the code and extract the constant quantity all here
    """
    def get_svm(self):
        return np.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])

    def get_knn(self):
        return np.array([1, 3, 5, 7, 9, 11, 13, 15])

    def get_gradientboosting(self):
        return np.array([10, 25, 50, 100, 200, 500])

    def get_randomforest(self):
        return np.array([10, 25, 50, 100, 200, 500])
    def get_music_type(self):
        return ['*.wav', '*.aif',  '*.aiff', '*.mp3', '*.au', '*.ogg']

    #Features that are related to the beat tracking task:
    def get_selected_features(self):
        selected_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15, 16, 17, 18]


class feature_reposyitory:
    def get_features(self):
        return ['zcr_mean', 'energy_mean', 'energy_entropy_mean', 'spectral_centroid_mean', 'spectral_spread_mean', 'spectral_entropy_mean', 'spectral_flux_mean', 'spectral_rolloff_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean',
                'mfcc_12_mean', 'mfcc_13_mean', 'chroma_1_mean', 'chroma_2_mean', 'chroma_3_mean', 'chroma_4_mean', 'chroma_5_mean', 'chroma_6_mean', 'chroma_7_mean', 'chroma_8_mean', 'chroma_9_mean', 'chroma_10_mean', 'chroma_11_mean', 'chroma_12_mean', 'chroma_std_mean', 'delta zcr_mean', 'delta energy_mean', 'delta energy_entropy_mean', 'delta spectral_centroid_mean',
                'delta spectral_spread_mean', 'delta spectral_entropy_mean', 'delta spectral_flux_mean', 'delta spectral_rolloff_mean', 'delta mfcc_1_mean', 'delta mfcc_2_mean', 'delta mfcc_3_mean', 'delta mfcc_4_mean', 'delta mfcc_5_mean', 'delta mfcc_6_mean', 'delta mfcc_7_mean', 'delta mfcc_8_mean', 'delta mfcc_9_mean', 'delta mfcc_10_mean', 'delta mfcc_11_mean', 'delta mfcc_12_mean', 'delta mfcc_13_mean',
                'delta chroma_1_mean', 'delta chroma_2_mean', 'delta chroma_3_mean', 'delta chroma_4_mean', 'delta chroma_5_mean', 'delta chroma_6_mean', 'delta chroma_7_mean', 'delta chroma_8_mean', 'delta chroma_9_mean', 'delta chroma_10_mean', 'delta chroma_11_mean', 'delta chroma_12_mean',
                'delta chroma_std_mean', 'zcr_std', 'energy_std', 'energy_entropy_std', 'spectral_centroid_std', 'spectral_spread_std', 'spectral_entropy_std', 'spectral_flux_std', 'spectral_rolloff_std', 'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 'mfcc_4_std', 'mfcc_5_std', 'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_std', 'mfcc_9_std', 'mfcc_10_std',
                'mfcc_11_std', 'mfcc_12_std', 'mfcc_13_std', 'chroma_1_std', 'chroma_2_std', 'chroma_3_std', 'chroma_4_std', 'chroma_5_std', 'chroma_6_std', 'chroma_7_std', 'chroma_8_std', 'chroma_9_std', 'chroma_10_std', 'chroma_11_std', 'chroma_12_std', 'chroma_std_std',
                'delta zcr_std', 'delta energy_std', 'delta energy_entropy_std', 'delta spectral_centroid_std', 'delta spectral_spread_std', 'delta spectral_entropy_std', 'delta spectral_flux_std', 'delta spectral_rolloff_std', 'delta mfcc_1_std', 'delta mfcc_2_std', 'delta mfcc_3_std', 'delta mfcc_4_std', 'delta mfcc_5_std', 'delta mfcc_6_std', 'delta mfcc_7_std', 'delta mfcc_8_std', 'delta mfcc_9_std', 'delta mfcc_10_std', 'delta mfcc_11_std',
                'delta mfcc_12_std', 'delta mfcc_13_std', 'delta chroma_1_std', 'delta chroma_2_std', 'delta chroma_3_std', 'delta chroma_4_std', 'delta chroma_5_std', 'delta chroma_6_std', 'delta chroma_7_std', 'delta chroma_8_std', 'delta chroma_9_std', 'delta chroma_10_std', 'delta chroma_11_std', 'delta chroma_12_std', 'delta chroma_std_std']