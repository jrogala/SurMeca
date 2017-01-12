import scipy.signal
import numpy
import tool

NPERSEGVALUE = 1000

def frf(i,o,est = "H1"):
    """
    Return the frf of input/output
    """
    ifft = numpy.fft.fft(i)
    offt = numpy.fft.fft(o)
    oicsd = scipy.signal.csd(o,i,nperseg = NPERSEGVALUE)[1]
    iwelch = scipy.signal.welch(i,nperseg = NPERSEGVALUE)[1]
    return (oicsd/iwelch)
