import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from scipy.stats import laplace
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def estimate_period(request_array, max_lag=2000):
    n = len(request_array)
    mean = np.mean(request_array)
    var = np.var(request_array)
    
    autocorr = np.correlate(request_array - mean, request_array - mean, mode='full') / (var * n)
    autocorr = autocorr[n:]  
    
    peaks, _ = find_peaks(autocorr[1:max_lag])  
    if len(peaks) == 0:
        return None 
    peak_lags = peaks[np.argsort(autocorr[peaks])[::-1]]  
    estimated_period = peak_lags[0]  

    return estimated_period

# def build_interval_generator(request_array, period:int = 1440):
#     t = np.arange(len(request_array))
    
#     stl = STL(request_array, period=period)
#     result = stl.fit()
#     workload_smooth = result.trend + result.seasonal
#     residual = result.resid

#     resid_params = laplace.fit(residual)

#     f = interp1d(t, workload_smooth, kind='linear', fill_value="extrapolate")

#     def g(t_input: float):
#         lam = f(t_input)
#         if lam <= 0:
#             lam = 0.0001
#         sampled_noise = laplace.rvs(*resid_params)
#         interval = max(1.0 / lam + sampled_noise / lam, 0)
#         interval = min(interval, 15)
#         return interval

#     return g

def build_interval_generator(request_array, period:int = 1440):
    t = np.arange(len(request_array))
    
    # stl = STL(request_array, period=period)
    # result = stl.fit()
    # workload_smooth = result.trend + result.seasonal
    # residual = result.resid

    # resid_params = laplace.fit(residual)

    f = interp1d(t, request_array, kind='linear', fill_value="extrapolate")

    def g(t_input: float):
        lam = f(t_input)
        if lam <= 0:
            lam = 0.0001
        interval = max(1.0 / lam, 0)
        interval = min(interval, 15)
        return interval

    return g


