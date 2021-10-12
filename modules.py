import numpy as np
import librosa
import filters
import scipy

def wah_wah(x, fs):
    """
    wah_wah.py   state variable band pass
    
    BP filter with narrow pass band, Fc oscillates up and down the spectrum
    Difference equation taken from DAFX chapter 2
    Changing this from a BP to a BS (notch instead of a bandpass) converts this effect to a phaser

    yl(n) = F1*yb(n) + yl(n-1)
    yb(n) = F1*yh(n) + yb(n-1)
    yh(n) = x(n) - yl(n-1) - Q1*yb(n-1)

    vary Fc from 500 to 5000 Hz
    44100 samples per sec
    """
    # damping factor
    # lower the damping factor the smaller the pass band
    damp = 0.05

    # min and max centre cutoff frequency of variable bandpass filter
    minf = 500
    maxf = 3000

    # wah frequency, how many Hz per second are cycled through
    fw = 2000 
    #########################################################################

    # change in centre frequency per sample (Hz)
    # delta=0.1
    delta = fw/fs
    #0.1 => at 44100 samples per second should mean  4.41kHz Fc shift per sec

    # create triangle wave of centre frequency values
    fc = np.arange(minf, maxf, delta)
    while len(fc) < len(x):
        fc = np.append(fc, np.arange(maxf, minf, -delta))
        fc = np.append(fc, np.arange(minf, maxf, delta))
    
    # trim tri wave to size of input
    fc = fc[1:len(x)]

    # difference equation coefficients
    F1 = 2 * np.sin((np.pi * fc[1])/fs)  # must be recalculated each time Fc changes
    Q1 = 2 * damp                        # this dictates size of the pass bands

    yh = np.zeros(x.shape[0])               # create emptly out vectors
    yb = np.zeros(x.shape[0])
    yl = np.zeros(x.shape[0])

    # first sample, to avoid referencing of negative signals
    yh[1] = x[1]
    yb[1] = F1 * yh[1]
    yl[1] = F1 * yb[1]

    # apply difference equation to the sample
    for n in range(2, len(x) - 1):
        yh[n] = x[n] - yl[n - 1] - Q1 * yb[n - 1]
        yb[n] = F1 * yh[n] + yb[n - 1]
        yl[n] = F1 * yb[n] + yl[n - 1]
        
        F1 = 2 * np.sin((np.pi * fc[n])/fs)

    #normalise
    maxyb = max(abs(yb))
    y = yb/maxyb

    return y

def fuzz(x, fs):
    # parameters to vary the effect #
    clip = 0.1  # clipping threshold 0-0.99
    #################################

    amp = 1 / clip   # amplify coefficient
    amp = amp - 1 # this ensures max output is 0.99 and min is -0.99. Otherwise output is clipped anyway, experiment with bigger amps eg.500, 1000...

    # load a WAVE file samples into wave. Amplitude values are in the range [-1,+1]
    # returns the sample rate (Fs) in Hertz used to encode the data
    # determine number of samples
    no_samples = x.shape[0]

    # for each sample (is there a more efficient way of doing this???)
    for i in range(no_samples):

        # clip at both ends
        if x[i] > clip:
            x[i] = clip 

        if x[i] < -clip:
            x[i] = -clip

        # amplify!
        x[i] = amp * x[i]

    y = x

    return y

def fuzzexp(x, fs):
    # Distortion based on an exponential function
    # x    - input
    # gain - amount of distortion, >0->
    # mix  - mix of original and distorted sound, 1=only distorted
    gain = 11
    mix = 1

    q = x * gain / max(abs(x))
    z = np.sign(-q) * (1 - np.exp(np.sign(-q) * q))
    y = mix * z * max(abs(x)) / max(abs(z)) + (1 - mix) * x
    y = y * max(abs(x)) / max(abs(y))

    return y


def overdrive(x, fs):
    """
    "Overdrive" simulation with symmetrical clipping
    x - input
    """

    N = len(x)
    y = np.zeros(N) #Preallocate y
    th = 1/3 #threshold for symmetrical soft clipping 
    
    #by Schetzen Formula
    for i in range(N):
        if abs(x[i]) < th:
            y[i] = 2 * x[i]

        if abs(x[i]) >= th:
            if x[i] > 0: 
                y[i] = 3 - (2 - x[i] * 3 ** 2) / 3
            if x[i] < 0:
                y[i] = -(3 - (2 - abs(x[i]) * 3) ** 2) / 3

        if abs(x[i]) > 2 * th:
            if x[i] > 0: 
                y[i] = 1
            if x[i] < 0:
                y[i] = -1

    return y

def delay(x, fs):
    """
    Python Script that creates a multiple delay effect using a loop
    Possibility for extension:
    - do calculations with sampling frequency to convert delay in samples into miliseconds (need 44.1kHz samples)
    - exchange amplitude co-efficients and number of delays for a deterioration rate and shape
    """
    # parameters to vary the effect #
    # number of samples first delay is offset
    # should be greater than 1000 to produce an audible effect
    samp_delay = 2000       

    zero_padding = 10 * samp_delay   # length in samples of silence added to end of in-sample to hear the delays
    no_delays = 10      # number of delays to apply max of 1/dim
    dim = 0.1      # diminishing amplitude between subsequent delays, deterioration rate

    #################################

    if no_delays > 1/dim :
        print('ERROR: no_delays must be less than or equal to 1/dim!')

    # if this is a stereo sample
    if x.shape[-1] == 2:
        x = x[:,1] # convert to mono (add this to all m files)

    x = np.expand_dims(x, axis=1)
    # print(x.shape, np.zeros((zero_padding, 1)).shape)
    x = np.concatenate((x, np.zeros((zero_padding, 1))), axis=0)    # add zero's to the end of the wave to hear trailing delays

    # print(x.shape)
    # y = np.zeros((len(x),1))
    # y[:len(x),:] = x   # create empty out vector, copy initial signal to an out signal
    y = x
    amp = [1 - j * dim for j in range(no_delays - 1)]  # calculate amplitudes

    # for each sample
    for i in range(samp_delay, len(x)):
        #  for each delay
        for j in range(no_delays - 1):
            if i > j * samp_delay:   # causality
                if amp[j] > 0:       # no multiplying by negative amplitudes
                    y[i] = y[i] + amp[j] * x[i - j * samp_delay]    # add a delayed diminished sample

    return y

def pan(x, fs):
    """
      script based on DAFZ p 140 to
      perform matrix based panning of mono sound to stereo
    """
    initial_angle = -40 #in degrees
    final_angle = 40    #in degrees
    segments = 32
    angle_increment = (initial_angle - final_angle) / segments * np.pi / 180 # in radians
    lenseg = int(np.floor(len(x) / segments) - 1)
    pointer = 1
    angle = initial_angle * np.pi / 180 #in radians

    y = np.array([])
    for i in range(segments):
        A = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        stereo_x = np.array((x[pointer:pointer + lenseg], x[pointer:pointer + lenseg]))
        y = np.concatenate((y, np.matmul(A, stereo_x)), axis=1) if y.size != 0 else np.matmul(A, stereo_x)
        angle = angle + angle_increment 
        pointer = pointer + lenseg

    return y.transpose()

def compressor(x, fs):
    """
    # comp	- compression: 0 > comp > -1, expansion: 0 < comp < 1
    # a	- filter parameter < 1
    """

    comp = -0.5 # set compressor
    a = 0.5

    h = scipy.signal.lfilter(np.array([(1 - a) ** 2]), np.array([1.0, -2 * a, a ** 2]), abs(x))
    h = h / max(h)
    h = h ** comp

    y = x * h
    y = y * max(abs(x)) / max(abs(y))

    return y

def schroeder1(x, fs):
    
    """
    The structure is:  [y,b,a] = schroeder1(x,n,g,d,k) 
    
    where x = the input signal
          n = the number of allpass filters	
          g = the gain of the allpass filters (this should be less than 1 for stability)
          d = a vector which contains the delay length of each allpass filter
          k = the gain factor of the direct signal
          y = the output signal
          b = the numerator coefficients of the transfer function
          a = the denominator coefficients of the transfer function
    
    note: Make sure that d is the same length as n.
    """

    # Set the number of allpass filters
    n = 6
    # Set the gain of the allpass filters
    g = 0.9
    # Set delay of each allpass filter in number of samples
    d = np.floor(0.05 * np.random.rand(n) * fs).astype(int)
    print(d)
    #set gain of direct signal
    k = 0.2

    # send the input signal through the first allpass filter
    y, b, a = filters.allpass(x, g, d[0])

    # send the output of each allpass filter to the input of the next allpass filter
    for i in range(1, n):
        y, b1, a1 = filters.allpass(y, g, d[i])
        b, a = filters.seriescoefficients(b1, a1, b, a)

    # add the scaled direct signal
    y = y + k * x

    # normalize the output signal
    y = y / max(y)

    return y

def moorer(x, fs):
    """
    This is a reverberator based on Moorer's design which consists of 6 parallel feedback comb filters 
    (each with a low pass filter in the feedback loop) in series with an all pass filter.
    
    The structure is:  [y,b,a] = moorer(x,cg,cg1,cd,ag,ad,k)
    
    where x = the input signal
          cg = a vector of length 6 which contains g2/(1-g1) (this should be less than 1 for stability),
               where g2 is the feedback gain of each of the comb filters and g1 is from the following parameter 
          cg1 = a vector of length 6 which contains the gain of the low pass filters in the feedback loop of
                each of the comb filters (should be less than 1 for stability)
          cd = a vector of length 6 which contains the delay of each of the comb filters 
          ag = the gain of the allpass filter (this should be less than 1 for stability)
          ad = the delay of the allpass filter 
          k = the gain factor of the direct signal
          y = the output signal
          b = the numerator coefficients of the transfer function
          a = the denominator coefficients of the transfer function
    """

    cd = np.floor(0.05 * np.random.rand(6) * fs).astype(int)

    # set gains of 6 comb pass filters
    g1 = 0.5 * np.ones(6)
    # set feedback of each comb filter
    g2 = 0.5 * np.ones(6)
    # set input cg and cg1 for moorer function see help moorer
    cg = g2 / (1 - g1)
    cg1 = g1

    # set gain of allpass filter
    ag = 0.7
    # set delay of allpass filter
    ad = 0.08 * fs
    # set direct signal gain
    k = 0.5

    # send the input to each of the 6 comb filters separately
    [outcomb1,b1,a1] = filters.lpcomb(x, cg(1), cg1(1), cd(1))
    [outcomb2,b2,a2] = filters.lpcomb(x, cg(2), cg1(2), cd(2))
    [outcomb3,b3,a3] = filters.lpcomb(x, cg(3), cg1(3), cd(3))
    [outcomb4,b4,a4] = filters.lpcomb(x, cg(4), cg1(4), cd(4))
    [outcomb5,b5,a5] = filters.lpcomb(x, cg(5), cg1(5), cd(5))
    [outcomb6,b6,a6] = filters.lpcomb(x, cg(6), cg1(6), cd(6))

    # sum the ouptut of the 6 comb filters
    apinput = outcomb1 + outcomb2 + outcomb3 + outcomb4 + outcomb5 + outcomb6 

    #find the combined filter coefficients of the the comb filters
    [b,a] = filters.parallelcoefficients(b1,a1,b2,a2)
    [b,a] = filters.parallelcoefficients(b,a,b3,a3)
    [b,a] = filters.parallelcoefficients(b,a,b4,a4)
    [b,a] = filters.parallelcoefficients(b,a,b5,a5)
    [b,a] = filters.parallelcoefficients(b,a,b6,a6)

    # send the output of the comb filters to the allpass filter
    [y,b7,a7] = filters.allpass(apinput, ag, ad)

    #find the combined filter coefficients of the the comb filters in series with the allpass filters
    [b,a] = filters.seriescoefficients(b,a,b7,a7)

    # add the scaled direct signal
    y = y + k * x

    # normalize the output signal
    y = y / max(y)

    return y

def reverb_conv(x, h):
    """
    FCONV Fast Convolution
    [y] = FCONV(x, h) convolves x and h, and normalizes the output to +-1.

    x = input vector
    h = input effect vector

    See also CONV

    NOTES:

    1) I have a short article explaining what a convolution is.  It
        is available at http://stevem.us/fconv.html.
    
    """

    Ly = len(x) + len(h) -1   
    Ly2 = np.ceil(np.log2(abs(Ly))) ** 2 # Find smallest power of 2 that is > Ly
    X = scipy.fft(x, Ly2)		         # Fast Fourier transform
    H = scipy.fft(h, Ly2)	             # Fast Fourier transform
    Y = X * H        	            
    y, phase = scipy.ifft(Y, Ly2)        # Inverse fast Fourier transform
    y = y[1:1:Ly]                        # Take just the first N elements
    y = y / max(abs(y))                  # Normalize the output

    return y 





    
