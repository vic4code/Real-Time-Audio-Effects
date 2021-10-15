import numpy as np
import scipy

Epsilon = 1e-4

def allpass(x, g, d):
    """
    This is an allpass filter function.
    
    The structure is:  [y,b,a] = allpass(x,g,d)
    
    where x = the input signal
          g = the feedforward gain (the feedback gain is the negative of this) (this should be less than 1 for stability)
          d = the delay length
          y = the output signal
          b = the numerator coefficients of the transfer function
          a = the denominator coefficients of the transfer function
    """

    # If the feedback gain is more than 1, set it to 0.7 .
    if g >= 1:
        g = 0.7

    # Set the b and a coefficients of the transfer function depending on g and d.
    b = np.concatenate((np.array([g]), np.zeros(d), np.array([1])))
    a = np.concatenate((np.array([1]), np.zeros(d), np.array([g])))

    # filter the input signal 
    y = scipy.signal.lfilter(b, a, x)

    return y, b, a

def fbcomb(x, g, d):
    """
    This is a feedback comb filter function.
    
    The structure is:  [y,b,a] = fbcomb(x,g,d)
    
    where x = the input signal
          g = the feedback gain (this should be less than 1 for stability)
          d = the delay length
          y = the output signal
          b = the numerator coefficients of the transfer function
          a = the denominator coefficients of the transfer function
    
    
     Gautham J. Mysore - gauthamjm@yahoo.com
    
    """

    # If the feedback gain is more than 1, set it to 0.7 .
    if g >= 1:
        g = 0.7  

    # Set the b and a coefficients of the transfer function depending on g and d.
    b = np.concatenate((np.zeros(d), np.array([1])))
    a = np.concatenate((np.array([1]), np.zeros(d - 1), np.array([-g])))

    # filter the input signal 
    y = scipy.signal.lfilter(b, a, x)

    return y, b, a

def lpcomb(x, g, g1, d):
    
    """
    This is a feedback comb filter with a low pass filter in the feedback.
    
    The structure is:  [y,b,a] = lpcomb(x,g,g1,d)
    
    where x = the input signal
          g = g2/(1-g1), where g2 is the feedback gain of the comb filter (this should be less than 1 for stability)
          g1 = the feedback gain of the low pass filter (this should be less than 1 for stability)
          d = the delay length
          y = the output signal
          b = the numerator coefficients of the transfer function
          a = the denominator coefficients of the transfer function
    """

    # If g is more than 1, set it to 0.7 .
    if g >= 1:
        g = 0.7

    # If the low pass feedback gain is more than 1, set it to 0.7 .
    if g1 >= 1:
        g1 = 0.7

    # Set the b and a coefficients of the transfer function depending on g, g1 and d.
    b = np.concatenate((np.zeros(d), np.array([1]), np.array([-g1])))
    a = np.concatenate((np.array([1]), np.array([-g1]), np.zeros(d - 2), np.array([-g * (1 - g1)])))

    # filter the input signal 
    y = scipy.signal.lfilter(b, a, x)

    return y, b, a


def seriescoefficients(b1, a1, b2, a2):
    """
    This function gives the filter coefficients of the series connection of two filters.
    
    The structure is:  [b,a] = seriescoefficients(b1,a1,b2,a2)
    
    where b1 = the numerator coefficients of the 1st transfer function
          a1 = the denominator coefficients of the 1st transfer function
          b2 = the numerator coefficients of the 2nd transfer function
          a2 = the denominator coefficients of the 2nd transfer function
          b = the numerator coefficients of the composite transfer function
          a = the denominator coefficients of the composite transfer function
    """
    b = np.convolve(b1, b2)
    a = np.convolve(a1, a2)

    return b, a

def parallelcoefficients(b1, a1, b2, a2):

    """
    This function gives the filter coefficients of the parallel connection of two filters.
    The structure is:  [b,a] = parallelcoefficients(b1,a1,b2,a2)

    where b1 = the numerator coefficients of the 1st transfer function
        a1 = the denominator coefficients of the 1st transfer function
        b2 = the numerator coefficients of the 2nd transfer function
        a2 = the denominator coefficients of the 2nd transfer function
        b = the numerator coefficients of the composite transfer function
        a = the denominator coefficients of the composite transfer function

    """
    b = np.convolve(b1, a2) + np.convolve(b2, a1)
    a = np.convolve(a1, a2)

    return b, a

# Do not use this one, it's only used in the next function!!!
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a

# Bandpass filter applied on array "data"
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

# Do not use!!!
def iir_butter(lowcut, highcut, order=5):
    return scipy.signal.iirfilter(N=order, Wn=[lowcut, highcut],  btype='band', ftype='butter', analog = False, output='ba')


# IIR Bandpass filter applied on array "data" between 0 and 1
def iir_butter_filter(data, lowcut, highcut, order=5):
    b, a = iir_butter(lowcut, highcut, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

# Moving average to scmooth signal
def smooth(x, window_len=11, window='hanning'):
	if window_len < 3:
	    return x
	s= np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
	if window == 'flat': #moving average
	    w=np.ones(window_len,'d')
	else:
	    w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(), s, mode='valid')
	return y

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def compute_envelope (x, ma_width = 2000):
    """
    Compute the envelope of the signal for a dynamic filter application. It is just a moving average applied to the absolute value of the signal,
    normalized between 0 and 1 with some flooring/ceiling.
    Parameters:
    x = The audio signal, between -1 and 1
    ma_width = The width of the moving average.
    Return:
    The envelope of the signal, between 0 and 1
    """

    # Compute the envelope
    x = np.squeeze(x)
    envelope = np.concatenate((np.zeros(ma_width//2), x, np.zeros(ma_width//2)))
    envelope = moving_average(np.abs(envelope), ma_width)
    # Log based auditory system
    envelope_log = np.exp(envelope)
    # Normalization
    envelope_log -= np.min(envelope_log)
    envelope_log /= np.max(envelope_log) + Epsilon
    # 
    envelopeF = envelope_log-np.percentile(envelope_log, 10)
    envelopeF /= np.percentile(envelope_log, 95) + Epsilon
    envelopeF = np.clip(envelopeF, 0, 1)
    envelopeF /= np.max(envelopeF) + Epsilon
    return envelopeF

