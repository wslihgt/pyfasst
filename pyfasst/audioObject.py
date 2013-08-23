"""AudioObject

The audioObject module provides a class for input/output of audio
WAV files.

Originally meant to wrap scikits.audiolab, yet allowing to load the
scipy.io.wavfile if audiolab_ is missing.

Unfortunately it got awfully complicated in time, and a clean up is
necessary. Notably, there are problems with scaling, and some issues
related to the type of the file from audiolab_. 

.. _audiolab: http://pypi.python.org/pypi/scikits.auidolab

Jean-Louis Durrieu, 2012 - 2013
"""


import numpy as np
import warnings

from tools.utils import *

"""
# functions to read and write audio files
#     note that if audiolab is available, there are going to be
#     more available audio formats to read/write.
"""
try:
    import scikits.audiolab as al
    """
    # import marcel # just to make it read with scipy
    """
    
    def wavread(filename, first=0, last=None):
        sndfile = al.pysndfile.Sndfile(filename, mode='r')
        if last is None:
            last = sndfile.nframes
        
        sndfile.seek(first)
        fs = sndfile.samplerate
        data = sndfile.read_frames(nframes=last-first, )
        # Note: read_frame(nframes,dtype=np.int16) is broken! probably... 
        if sndfile.encoding == 'pcm16':
            # convert back to signed int16
            data = np.int16(data * 2.**15)
        sndfile.close()
        return fs, data, sndfile.encoding
    
    def wavwrite(filename, rate, data,
                 formattype='wav',
                 formatenc='pcm16',
                 formatend='file'):
        if formattype not in al.pysndfile.available_file_formats():
            raise AttributeError(formattype+' not available in sndfile.')
        if formatenc not in al.pysndfile.available_encodings(formattype):
            formatenc = al.pysndfile.available_encodings(formattype)[0]
        format = al.pysndfile.Format(type=formattype,
                                     encoding=formatenc,
                                     endianness=formatend)
        sndfile = al.pysndfile.Sndfile(filename=filename,
                                       mode='w',
                                       samplerate=rate,
                                       format=format,
                                       channels=data.shape[1]
                                       )
        sndfile.write_frames(np.int16(data))
        sndfile.close()
        return 0
    
    print "Reading and Writing WAV files with audiolab"
except ImportError:
    print "Using scipy.io.wavfile"
    import scipy.io.wavfile as wav
    
    def wavread(filename, first=0, last=None):
        fs, data = wav.read(filename)
        data = data[first:last]
        encoding = data.dtype
        
        return fs, data, encoding
    
    def wavwrite(filename, rate, data,
                 formattype='wav',
                 formatenc='int16',
                 formatend='file'):
        if formatenc not in ('int16', 'int32', 'int8'):
            if np.abs(data).max()>2**15:
                formatenc = 'int32'
            elif np.abs(data).max()>2**7:
                formatenc = 'int16'
            else:
                formatenc = 'int8'
            print "Changing encoding to", formatenc
        data_ = np.array(data, dtype=formatenc)
        # print data_.dtype
        wav.write(filename, rate, data_)
        return 0

class AudioObject(object):
    """A wrapper for the wrapper by D. Cournapeau. Or in case it is not
    installed, it falls back on :py:mod:`scipy.io.wavfile`.
    """
    def __init__(self, filename, mode='rw'):
        """AudioObject initialization
        
        
        """
        self.filename = filename
        self.mode = mode
    
    def _read(self):
        if 'r' not in self.mode:
            raise ValueError("Not in read mode.")
        self._samplerate, self._data, self._encoding = (
            wavread(self.filename))
        if len(self._data.shape)==2:
            self._nframes, self._channels = self._data.shape
        else:
            self._nframes = self._data.size
            self._channels = 1
        
        # rescaling the data array
        self._maxdata = np.maximum(
            1.1 * np.abs(self._data).max(),
            1e-10)
        self._data = self._data / self._maxdata 
        # self._encoding = 
    
    def _write(self):
        if 'w' not in self.mode:
            raise ValueError("Not in write mode.")
        
        if 'w' in self.mode and not hasattr(self, '_samplerate') \
               and not hasattr(self, '_data'):
            raise AttributeError("Should set sample rate and have data "+\
                                 "in write mode.")
        
        ##print "Setting encoding to pcm16, converting data too..."
        ##self._encoding = 'pcm16'
        ##self._data = np.int16(self._maxdata * self._data)
        ##self._maxdata = 1
        
        wavwrite(filename=self.filename,
                 rate=self._samplerate,
                 data=self._maxdata * self._data,
                 formatenc=self._encoding)
    
    def _set_data(self, data):
        s = data.shape
        if s[0] < s[1] and s[1] > 2:
            print "Data shape is strangely ordered: transposing input data."
            self._data = np.array(data.T, order='C')
        else:
            self._data = np.array(data, order='C')
        self._maxdata = 1.1 * np.abs(self._data).max()
        self._encoding = self._data.dtype.name
        
        self._data = self._data / self._maxdata
    
    def _get_data(self):
        if not hasattr(self, '_data'):
            self._read()
        return self._data
    
    def _del_data(self):
        if hasattr(self, 'data'):
            del self._data
    
    data = property(_get_data, _set_data, _del_data)
    
    def _get_samplerate(self):
        if not hasattr(self, '_samplerate') and 'r' in self.mode:
            self._read()
        return self._samplerate
    
    def _set_samplerate(self, samplerate):
        if 'r' in self.mode:
            warnings.warn("Changing the sampling rate in read mode")
        self._samplerate = int(samplerate)
        
    samplerate = property(_get_samplerate, _set_samplerate)
    fs = samplerate # just an alias, for ease of use
    
    @property 
    def channels(self):
        if not hasattr(self, '_channels'):
            self._read()
        return self._channels
    
    @property 
    def nframes(self):
        if not hasattr(self, '_nframes'):
            self._read()
        return self._nframes
