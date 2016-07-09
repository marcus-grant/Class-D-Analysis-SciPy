#!/usr/bin/env Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def triangle_wave(time_samples, min_value, max_value, bit_depth):
    tri_samples = np.zeros(len(time_samples))
    signal_range = np.float(abs(max_value - min_value))
    print("SignalRange = " + str(signal_range) + ", discreteValues = " + str(2 ** bit_depth))
    dx = 4 * signal_range / (2 ** bit_depth)
    print("dx = " + str(dx))
    iterator = 0
    x = min_value
    while iterator < len(time_samples):
        tri_samples[iterator] = x
        x += dx
        if (x > max_value) or (x < min_value):
            dx *= -1
            x = x + 2*dx
            tri_samples[iterator] = x
        iterator += 1
        # print(x)
    return tri_samples

# def triangle_wave(time_samples, frequency, min_value, max_value):
#     output_samples = np.zeros(len(time_samples))
#     output_range = max_value - min_value
#     dt = time_samples[1] - time_samples[0]
#     dout = 2 * frequency * output_range / dt
#     i = 0
#     current_sample = min_value
#     while i < len(time_samples):
#         if current_sample + dout > max_value:
#             current_sample = max_value
#             dout *= -1
#         elif current_sample + dout < min_value:
#             current_sample = min_value
#             dout *= -1
#         else:
#             current_sample += dout
#         output_samples[i] = current_sample
#         i += 1
#     return output_samples


def comparator(sig_a, sig_b, min_value, max_value):
    sig_c = np.copy(sig_a)
    i = 0
    while i < sig_c.size:
        if sig_b[i] >= sig_a[i]:
            sig_c[i] = max_value
        else:
            sig_c[i] = min_value
        i += 1
    return sig_c


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def low_pass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def quantize(data, bit_depth, min_value, max_value):
    quantized_data = np.copy(data)
    data_range = max_value - min_value
    quant_interaval = data_range/(2 ** bit_depth)
    i = 0
    while i < len(data):
        x = quantized_data[i]
        xq = np.floor(x / quant_interaval) * quant_interaval
        quantized_data[i] = xq
        i += 1
    print("len(quantData) = " + str(len(quantized_data)))
    return quantized_data

# analysis parameters

sample_freq = 60 * (10 ** 6)  # ADC sample frequency
sample_time = sample_freq ** -1
bit_depth = 10
start_time = 0
end_time = 1.5 / 1000
min_signal = -1
max_signal = 1
pi2 = np.pi * 2
f20hz = 20 * pi2
f50hz = 50 * pi2
f100hz = 100 * pi2
f200hz = 2 * f100hz
f500hz = 5 * f100hz
f1khz = 2 * f500hz
f2khz = 2 * f1khz
f5khz = 5 * f1khz
f10khz = 2 * f5khz
f20khz = 2 * f10khz
time_samples = np.arange(start_time, end_time, sample_time)
N = len(time_samples)
print("number of timeSamples: " + str(len(time_samples)))
triangle_samples = triangle_wave(time_samples, min_signal, max_signal, 10)

# test signal - sine wave
fSig = 15*(10**3)
fSigRads = fSig*np.pi*2
input_sig = .14 * np.sin(f20hz * time_samples) + .14 * np.sin(f50hz * time_samples) +\
            .14 * np.sin(f100hz * time_samples) + .14 * np.sin(f200hz * time_samples) +\
            .14 * np.sin(f500hz * time_samples) + .14 * np.sin(f1khz * time_samples) +\
            .14 * np.sin(f2khz * time_samples) + .14 * np.sin(f5khz * time_samples) +\
            .14 * np.sin(f10khz * time_samples) + .14 * np.sin(f20khz * time_samples)

quantized_input = quantize(input_sig, 10, np.min(input_sig), np.max(input_sig))
modulated_sig = comparator(triangle_samples, quantized_input, -1, 1)
output_sig = low_pass_filter(modulated_sig, 24000, 96000000, order=3)


###########################################################################################################
# Fourier transform stuff
# k = np.arange(N)
# frq = k / sample_time
# frq = frq[range(N/2)]
# input_fft = np.fft.fft(input_sig) / N
# input_fft = input_fft[range(N / 2)]
# output_fft = np.fft.fft(output_sig) / N
# output_fft = output_fft[range(N / 2)]
# print ("len(k) = " + str(len(k)) + "len(frq) = " + str(len(frq)) + "len(inFFT) = " + str(len(input_fft)))


###########################################################################################################
# plotting stuff
plt.close('all')
# plot triangle wave and quantized input signal
plt.subplot(3, 1, 1)
plt.scatter(time_samples, triangle_samples, color='b', alpha=0.3, marker='.')
plt.scatter(time_samples, quantized_input, color='r', alpha=0.3, marker='.')
plt.scatter(time_samples, modulated_sig, color='g', alpha=0.3, marker='.')
plt.xlim(start_time, end_time)
plt.xticks(np.arange(start_time, end_time, (end_time - start_time) / 4))
plt.title('Quantized Input & Triangle Wave')
plt.ylabel('Sample Value')
# plot resulting PWM signal and its filtered output i.e. the final output signal
plt.subplot(3, 1, 2)
plt.plot(time_samples, quantized_input, color='b', alpha=0.3)
plt.plot(time_samples, output_sig, color='r', alpha=0.5)
plt.xlabel(' time (s)')
plt.ylabel('Output from modulater & output filter')


# configure x axis
plt.xlim(start_time, end_time)
plt.xticks(np.arange(start_time, end_time, (end_time - start_time) / 4))

# configure y axis
plt.ylim(-1.2,1.2)
plt.yticks([-1.0,0,1.0])

# plot fourier transform
plt.subplot(3, 1, 3)
plt.plot(frq, abs(input_fft),'r')
plt.plot(frq, abs(output_fft),'b')

plt.show()
