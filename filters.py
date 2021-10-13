import numpy as np
import scipy

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
    b = np.concatenate((np.array([g]), np.zeros(d - 1), np.array([1])))
    a = np.concatenate((np.array([1]), np.zeros(d - 1), np.array([g])))

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

