import numpy as np
import tensorflow as tf
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.signal.window_ops import _raised_cosine_window


def povey_window(window_length, periodic=True, dtype=dtypes.float32,
                   name=None):
    return tf.math.pow(_raised_cosine_window(name, 'hamming_window', window_length, periodic,
                               dtype, 0.5, 0.5), 0.85)

# def preemphasize(window, coef=0.97):
#     if(0 <= coef >= 1):
#         return window
#     orig_shape = len(window)
#     shifted_window = tf.concat([window, [window[0]]], axis=0)
#     rolled_window = tf.roll(shifted_window, shift=1, axis=0)
#     emphasized_window = shifted_window - coef * rolled_window
#     emphasized_window = emphasized_window[:orig_shape]
#     return emphasized_window

def stft(signals, fft_length, frame_step, frame_length,
         window='povey', name="espnet_stft"):

    if(window == 'povey'):
        window_fn = povey_window
    elif(window == 'hamming'):
        window_fn = window_ops.hamming_window
    else:
        window_fn = window_ops.hann_window

    with ops.name_scope(name, 'stft', [signals, frame_length,
                                     frame_step]):
        signals = ops.convert_to_tensor(signals, name='signals')
        signals.shape.with_rank_at_least(1)
        frame_length = ops.convert_to_tensor(frame_length, name='frame_length')
        frame_length.shape.assert_has_rank(0)
        frame_step = ops.convert_to_tensor(frame_step, name='frame_step')
        frame_step.shape.assert_has_rank(0)

        if fft_length is None:
            fft_length = _enclosing_power_of_two(frame_length)
        else:
            fft_length = ops.convert_to_tensor(fft_length, name='fft_length')

        framed_signals = shape_ops.frame(
            signals, frame_length, frame_step, pad_end=False)
        
        means = tf.reduce_mean(framed_signals, axis=1, keepdims=True)
        normalized_signals = framed_signals - means
        
        # emphasized_signals = tf.map_fn(preemphasize, normalized_signals, dtype=tf.float32, back_prop=False)
        
        # Optionally window the framed signals.
        if window_fn is not None:
            window = window_fn(frame_length, dtype=framed_signals.dtype)
            normalized_signals *= window

        # fft_ops.rfft produces the (fft_length/2 + 1) unique components of the
        # FFT of the real windowed signals in framed_signals.
    return fft_ops.rfft(normalized_signals, [fft_length])

def istft(x, n_shift, win_length=None, window='hann', center=True):
    # x: [Time, Channel, Freq]
    if x.ndim == 2:
        single_channel = True
        # x: [Time, Freq] -> [Time, Channel, Freq]
        x = x[:, None, :]
    else:
        single_channel = False

    # x: [Time, Channel]
    x = np.stack([librosa.istft(
        x[:, ch].T,  # [Time, Freq] -> [Freq, Time]
        hop_length=n_shift,
        win_length=win_length,
        window=window,
        center=center)
        for ch in range(x.shape[1])], axis=1)

    if single_channel:
        # x: [Time, Channel] -> [Time]
        x = x[:, 0]
    return x


def stft2logmelspectrogram(x_stft, fs, n_mels, n_fft, fmin=None, fmax=None,
                           eps=1e-10):
    # x_stft: (Time, Channel, Freq) or (Time, Freq)
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax

    # spc: (Time, Channel, Freq) or (Time, Freq)
    spc = np.abs(x_stft)
    # mel_basis: (Mel_freq, Freq)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    # lmspc: (Time, Channel, Mel_freq) or (Time, Mel_freq)
    lmspc = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    return lmspc

def get_log_mel_spectrogram(stfts, sampling_rate, num_mel_bins=80, fft_length = 512, fmin=20, fmax=None, eps=1e-10):
    
    spectrograms = tf.abs(stfts)
    
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz = 0 if fmin is None else fmin
    upper_edge_hertz = sampling_rate / 2 if fmax is None else fmax

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sampling_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))
    
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + eps)
    
    return log_mel_spectrograms.numpy()

def spectrogram(x, n_fft, n_shift, win_length=None, window='hann'):
    # x: (Time, Channel) -> spc: (Time, Channel, Freq)
    spc = np.abs(stft(x, n_fft, n_shift, win_length, window=window))
    return spc


def logmelspectrogram(x, fs, n_mels, n_fft, n_shift,
                      win_length=None, window='hann', fmin=None, fmax=None,
                      eps=1e-10, pad_mode='reflect'):
    # stft: (Time, Channel, Freq) or (Time, Freq)
    with tf.device("cpu:0"):
        x_stft = stft(x, fft_length=n_fft, frame_step=n_shift, frame_length=win_length,
                  window=window)
        log_mels = get_log_mel_spectrogram(stfts=x_stft, sampling_rate=fs, num_mel_bins=n_mels, fft_length=n_fft,fmin=fmin, fmax=fmax, eps=eps)
    
    return log_mels

class Spectrogram(object):
    def __init__(self, n_fft, n_shift, win_length=None, window='hann'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window))

    def __call__(self, x):
        return spectrogram(x,
                           n_fft=self.n_fft, n_shift=self.n_shift,
                           win_length=self.win_length,
                           window=self.window)


class LogMelSpectrogram(object):
    def __init__(self, fs, n_mels, n_fft, n_shift, win_length=None,
                 window='hann', fmin=None, fmax=None, eps=1e-10):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ('{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, '
                'n_shift={n_shift}, win_length={win_length}, window={window}, '
                'fmin={fmin}, fmax={fmax}, eps={eps}))'
                .format(name=self.__class__.__name__,
                        fs=self.fs,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        eps=self.eps))

    def __call__(self, x):
        return logmelspectrogram(
            x,
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft, n_shift=self.n_shift,
            win_length=self.win_length,
            window=self.window)


class Stft2LogMelSpectrogram(object):
    def __init__(self, fs, n_mels, n_fft, fmin=None, fmax=None, eps=1e-10):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ('{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, '
                'fmin={fmin}, fmax={fmax}, eps={eps}))'
                .format(name=self.__class__.__name__,
                        fs=self.fs,
                        n_mels=self.n_mels,
                        n_fft=self.n_fft,
                        fmin=self.fmin,
                        fmax=self.fmax,
                        eps=self.eps))

    def __call__(self, x):
        return stft2logmelspectrogram(
            x,
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax)


class Stft(object):
    def __init__(self, n_fft, n_shift, win_length=None,
                 window='hann', center=True, pad_mode='reflect'):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def __repr__(self):
        return ('{name}(n_fft={n_fft}, n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'center={center}, pad_mode={pad_mode})'
                .format(name=self.__class__.__name__,
                        n_fft=self.n_fft,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        center=self.center,
                        pad_mode=self.pad_mode))

    def __call__(self, x):
        return stft(x, self.n_fft, self.n_shift,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode=self.pad_mode)


class IStft(object):
    def __init__(self, n_shift, win_length=None, window='hann', center=True):
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center

    def __repr__(self):
        return ('{name}(n_shift={n_shift}, '
                'win_length={win_length}, window={window},'
                'center={center})'
                .format(name=self.__class__.__name__,
                        n_shift=self.n_shift,
                        win_length=self.win_length,
                        window=self.window,
                        center=self.center))

    def __call__(self, x):
        return istft(x, self.n_shift,
                     win_length=self.win_length,
                     window=self.window,
                     center=self.center)
