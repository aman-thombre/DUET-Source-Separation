import numpy as np
import scipy.io.wavfile
import os
from scipy import signal
from sklearn.cluster import KMeans

def duet_source_separation(mic_data_folder, NUM_SOURCES):
    '''
    DUET source separation algorithm.

    INPUTS:
        mic_data_folder: name of folder (without a trailing slash) containing 
                         two mic datafiles `0.wav` and `1.wav`.
        NUM_SOURCES: number of sources in the recording

    OUTPUTS:
        output: NUM_SOURCES * recording_length numpy array, containing the source-separated
                recordings
    '''
    
    ### GET IN DATA
    file0 = os.path.join(mic_data_folder, "0.wav")
    file1 = os.path.join(mic_data_folder, "1.wav")
    
    srate,wav0 = scipy.io.wavfile.read(file0) # sample rate, first waveform
    _,wav1 = scipy.io.wavfile.read(file1) # second waveform
    
    #wav0 = signal.savgol_filter(wav0,21,2)
    #wav1 = signal.savgol_filter(wav1,21,2)
    
    ### STFT
    f0,t0,Zxx0 = signal.stft(wav0,nperseg=2047) # get STFTs of both recordings
    f1,t1,Zxx1 = signal.stft(wav1,nperseg=2047)
    
    ### DELAY
    # Need to get frequency matrix so we can divide and get true delays
    # This is taken from the 2007 - Springer: "The DUET blind source separation algorithm" paper referenced above
    freq = ((np.concatenate((np.arange(1,513),np.arange(-512,0))))*((2*3.14159265)/1024)).reshape(-1,1)
    fmat = np.ones((Zxx1.shape[1],freq.shape[0])).T
    fmat *= freq

    ratio = (Zxx0+1e-32)/(Zxx1+1e-32) # get ratio of STFT so we can get delay
    delay = (np.imag(np.log(ratio))/fmat).reshape(-1,1) # get delay from imaginary, reshape so we can use KMeans
    
    #delay = (np.imag(np.log(ratio))/fmat).flatten() # delay and attenuation
    #atten = np.real(ratio).flatten()
    
    ### CLUSTERING
    kmeans = KMeans(n_clusters=NUM_SOURCES) # cluster object
    kmeans.fit_predict(delay) # clustering
    
    #cluster_data = np.array([[atten[i],delay[i]] for i in range(len(atten))]) # for the case of delay and attenuation
    #kmeans.fit_predict(cluster_data)
    
    labels = kmeans.labels_.reshape(ratio.shape) # put our labels back into the right shape
    
    ### OUTPUT
    output = np.zeros((NUM_SOURCES,wav0.shape[0]))

    for i in range(NUM_SOURCES):
        f = Zxx0 * (labels==i) # isolate sources
        t,x = signal.istft(f) # inverse transform
        output[i,:] = signal.savgol_filter(x,11,2)[:wav0.shape[0]] # put in smoothed output
        #output[i,:] = x[:wav0.shape[0]]

    return output.astype(np.int16)