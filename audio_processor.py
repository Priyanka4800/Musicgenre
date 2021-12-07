import librosa.display
import numpy as np
import librosa as lr
from math import floor
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
from PIL import Image
from matplotlib.pyplot import imshow
# %matplotlib inline


def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    '''

   # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    audio_path=r'C:\Users\priya\Downloads\Arctic Monkeys - Do I Wanna Know (Official Video).wav'
    src, sr = lr.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    print(n_sample_fit)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    logam = lr.amplitude_to_db
    melgram = lr.feature.melspectrogram
    ret =logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS)**2)
    # ret = ret[np.newaxis, np.newaxis, :]
    log_power = lr.power_to_db(ret, ref=np.max)


    lr.display.specshow(ret, x_axis='time', y_axis='mel', cmap=cm.jet)
    plt.colorbar()
    return ret


def compute_melgram_multiframe(audio_path, all_song=True):
    ''' Compute a mel-spectrogram in multiple frames of the song and returns it in a shape of (N,1,96,1366), where
    96 == #mel-bins, 1366 == #time frame, and N=#frames

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..
    all_song=True
    if all_song:
        DURA_TRASH = 0
    else:
        DURA_TRASH = 20

    src, sr = lr.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    n_sample_trash = int(DURA_TRASH*SR)

    #remove the trash at the beginning and at the end
    src = src[n_sample_trash:(n_sample-n_sample_trash)]
    n_sample=n_sample-2*n_sample_trash


    # print n_sample
    # print n_sample_fit

    ret = np.zeros((0, 1, 96, 1366), dtype=np.float32)


    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
        logam = lr.amplitude_to_db
        melgram = lr.feature.melspectrogram
        ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS)**2)
        ret = ret[np.newaxis, np.newaxis, :]

    elif n_sample > n_sample_fit:  # if too long
        N=int(floor(n_sample/n_sample_fit))

        src_total=src
        print(N)
        for i in range(0,N):
            src = src_total[(i*n_sample_fit):(i+1)*(n_sample_fit)]

            logam = lr.amplitude_to_db
            melgram = lr.feature.melspectrogram
            retI = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                n_fft=N_FFT, n_mels=N_MELS)**2)
            
            log_power = lr.power_to_db(retI, ref=np.max)
            
            

            #lr.display.specshow(retI, x_axis='time', y_axis='mel', cmap=cm.jet)
            # plt.show()
            retI = retI[np.newaxis, np.newaxis, :]
            print("\n")

            #print retI.shape

            ret = np.concatenate((ret, retI), axis=0)

    return ret