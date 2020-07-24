
import matplotlib.pyplot as plt
import step_params
import chart_studio.plotly as py
from plotly import tools
from plotly.graph_objs import *
import plotly.express as px
import pandas as pd
import pyAudioAnalysis as audio

def generateFeatures_Single(filePath,visualization_feature_names):
    """
    This function is used to plot some feature of the music
    You could use it like this ().generateFeatures_Single("F:\\tempFile\pythonWav\sample1.wav",["zcr","chroma"])
    :return: Nothing at all
    """
    [Fs, x] = audio.audioBasicIO.read_audio_file(filePath)
    # 先合并成单声道
    x=audio.audioBasicIO.stereo_to_mono(x)
    # F 是n*...的，一行是一个feature
    F, f_names = audio.ShortTermFeatures.feature_extraction(x, Fs, 0.50*Fs, 0.25*Fs)

    pic=0
    t=[]
    for need in visualization_feature_names:
        for i,name in enumerate(f_names):
            if name.find(need)!=-1:
                pic+=1
                t.append(i)

    for p,i in enumerate(t):
        plt.plot(F[i, :]);
        plt.xlabel('Frame no');
        plt.ylabel(f_names[i])
        plt.show()


def generate_CompareGraph():

    [Fs, x] = audio.audioBasicIO.read_audio_file("F:\\tempFile\pythonWav\people\sample2.mp3")
    # 先合并成单声道
    x = audio.audioBasicIO.stereo_to_mono(x)
    # F 是n*...的，一行是一个feature
    F,_= audio.ShortTermFeatures.feature_extraction(x, Fs, 0.50 * Fs, 0.25 * Fs)




    frame=pd.DataFrame(F)
    frame.head()
    fig = px.line(frame)
    fig.show()


if __name__ == '__main__':
    generate_CompareGraph()