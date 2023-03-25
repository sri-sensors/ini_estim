from scipy.signal import butter, lfilter, filtfilt


def butter_lpf(f_high, f_sample, order=2):
    b, a = butter(order, 2*f_high/f_sample, btype='lowpass', output='ba')
    return b, a


def apply_butter_lpf(data, f_high, f_sample, order=2):
    b, a, = butter_lpf(f_high, f_sample, order)
    return lfilter(b, a, data)


def apply_butter_lpf_sym(data, f_high, f_sample, order=2):
    b, a, = butter_lpf(f_high, f_sample, order)
    return filtfilt(b, a, data, padtype='constant')
