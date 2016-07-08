#!/usr/bin/env Python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplt
from scipy.signal import butter, lfilter, freqz


def triangleWave(timeSamples, min, max, bitDepth):
    triSamples = np.zeros(len(timeSamples))
    signalRange = np.float(abs(max - min))
    print("SignalRange = "+str(signalRange)+", discreteValues = "+str(2**bitDepth))
    dx = 32 * signalRange / ( 2**bitDepth )
    print("dx = " + str(dx))
    iterator = 0
    x = min
    dt = timeSamples[1] - timeSamples[0]
    while iterator < len(timeSamples):
        triSamples[iterator] = x
        x += dx
        if (x > max) or (x < min):
            dx *= -1
            x = x + 2*dx
            triSamples[iterator] = x
        iterator += 1
        #print(x)
    return triSamples

def comparator(sigA, sigB, min, max):
    sigC = np.copy(sigA)
    i = 0
    while i < len(sigC):
        if sigB[i] >= sigA[i]:
            sigC[i] = max
        else:
            sigC[i] = min
        i += 1
    return sigC

def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowPassFilter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def adcSample(data, bitDepth):
    signalRange = np.float(abs(np.max(data) - np.min(data)))
    signalUInt = np.uint()
    resolution = signalRange / ( 2**bitDepth )
    sampledData = np.copy(data)
    sampledData = roundsampledData
    

#analysis parameters

fSample = 8*10**6 #ADC sample frequency of the ATMega328
tSample = fSample**-1
bitDepth = 10
startTime = 0
endTime = 0.001
minSignal = -1
maxSignal = 1
timeSamples = np.arange(startTime, endTime, tSample)
print("number of timeSamples: " + str(len(timeSamples)))
triangleSamples = triangleWave(timeSamples, minSignal, maxSignal, bitDepth)

#test signal - sine wave
fSig = 1*(10**2) #20kHz
fSigRads = fSig*np.pi*2
inputSig = np.sin(fSigRads*timeSamples)

modulatedSig = comparator(triangleSamples, inputSig, 0, 5)
outputSig = lowPassFilter(modulatedSig, 100, 96000, order=5)
  
#plotting stuff
#plot data
#plt.plot(timeSamples, triangleSamples, color='g', alpha=0.2)
#plt.plot(timeSamples, inputSig, color='b', alpha=0.3)
plt.plot(timeSamples, modulatedSig, color='b', alpha=0.3)
plt.plot(timeSamples, outputSig, color='r', alpha=0.5)

#configure colors

#configure x axis
plt.xlim(startTime,endTime)
plt.xticks(np.arange(startTime,endTime,(endTime - startTime)/4))

#configure y axis
plt.ylim(0,6)
plt.yticks([-1,0,1])

plt.show()
