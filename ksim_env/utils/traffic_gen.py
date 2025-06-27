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

def build_rpm(profile: pd.DataFrame, HashFunction: str, sim_duration: int) -> tuple:
    data = profile[profile['HashFunction'].str.startswith(HashFunction)]
    row = data.sample(n=1, random_state=42).iloc[0, 4:].values 
    period = 1440
    
    if len(row) < sim_duration:
        sim_duration = len(row)
        
    def randint_step(low, high, step):
        num = (high - low) // step
        return low + np.random.randint(0, num) * step
    
    # chọn lấy 1 ngày ngẫu nhiên
    start = randint_step(0, len(row) - sim_duration + 1, period) 
    
    # Nhân 10 lên mô phỏng vô hạn
    rpm = np.tile(row[start : start + sim_duration], 10)
    
    # period = estimate_period(rpm)
    return rpm, period, start 

def azure_ia_generator(rpm, period: int):
    ia_gen = build_interval_generator(rpm, period)
    
    t_now = 0.0 # phút
    while t_now < len(rpm):
        ia =  ia_gen(t_now)
        t_now += ia
        yield ia*60 # trả về theo đơn vị s


