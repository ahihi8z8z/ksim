import numpy as np

class StreamingStat:
    def __init__(self, hist_min=0.0, hist_max=1.0, hist_bins=20, max_quantile_samples=10_000):
        self.count = 0
        self.count_nonzero = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')

        # Histogram setup
        self.hist_min = hist_min
        self.hist_max = hist_max
        self.hist_bins = np.linspace(hist_min, hist_max, hist_bins + 1)
        self.hist_counts = np.zeros(hist_bins, dtype=int)

        # For quantile estimation
        self.max_quantile_samples = max_quantile_samples
        self.values_for_quantile = []

    def add(self, x: float):
        self.count += 1
        if x != 0:
            self.count_nonzero += 1
        self.sum += x
        self.sum_sq += x * x
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

        # Update histogram
        if self.hist_min <= x <= self.hist_max:
            bin_idx = np.searchsorted(self.hist_bins, x, side='right') - 1
            if 0 <= bin_idx < len(self.hist_counts):
                self.hist_counts[bin_idx] += 1

        # Store for percentile estimate (subsample)
        if len(self.values_for_quantile) < self.max_quantile_samples:
            self.values_for_quantile.append(x)

    def mean(self):
        return self.sum / self.count if self.count else 0.0

    def std(self):
        if self.count < 2:
            return 0.0
        mean_sq = self.sum * self.sum / self.count
        return np.sqrt((self.sum_sq - mean_sq) / (self.count - 1))

    def summary(self):
        quantiles = {}
        if self.values_for_quantile:
            vals = np.array(self.values_for_quantile)
            quantiles = {
                f'p{p}': float(np.percentile(vals, p)) for p in [50, 90, 95, 99]
            }

        return {
            'count': self.count,
            'nonzero': self.count_nonzero,
            'mean': self.mean(),
            'std': self.std(),
            'min': self.min_val,
            'max': self.max_val,
            'hist_bins': self.hist_bins.tolist(),
            'hist_counts': self.hist_counts.tolist(),
            **quantiles,
        }
