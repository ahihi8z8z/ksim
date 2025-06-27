from scipy.interpolate import PchipInterpolator
import numpy as np

def build_et_df(profile, HashFunction):
    filtered = profile[profile['HashFunction'].str.startswith(HashFunction)].copy()
    filtered = filtered.reset_index(drop=True)
    return filtered

def azure_et_generator(et_df, start_day):
    print(f"aaaaaaaaaaaaaaaaaaaaaa {start_day}")
    row = et_df.iloc[start_day]

    percentile_columns = {
        0: "percentile_Average_0",
        1: "percentile_Average_1",
        25: "percentile_Average_25",
        50: "percentile_Average_50",
        75: "percentile_Average_75",
        99: "percentile_Average_99",
        100: "percentile_Average_100"
    }
    percentiles = sorted(percentile_columns.keys())
    xs = [row[percentile_columns[p]] for p in percentiles]
    ps = [p / 100.0 for p in percentiles]

    interpolator = PchipInterpolator(xs, ps, extrapolate=True)
    x_range = np.linspace(row["Minimum"], row["Maximum"], 1000)
    cdf_vals = np.clip(interpolator(x_range), 0, 1)

    sorted_idx = np.argsort(x_range)
    x_sorted = x_range[sorted_idx]
    cdf_sorted = cdf_vals[sorted_idx]

    cdf_clean, unique_indices = np.unique(cdf_sorted, return_index=True)
    x_clean = x_sorted[unique_indices]

    if len(cdf_clean) < 4:
        raise ValueError(f"Day {start_day} has insufficient unique CDF points.")

    inverse_cdf = PchipInterpolator(cdf_clean, x_clean, extrapolate=True)

    def sample_one():
        # Trả về đơn vị giây
        return inverse_cdf(np.random.uniform(0, 1)) / 1000


    return sample_one