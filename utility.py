import time
import numpy as np
import puzzlepiece as pzp


# Utility class to fetch image and intensity from camera
class CameraImageFetcher:
    def __init__(self, puzzle: pzp.Puzzle, wait_time: float=0):
        self.puzzle = puzzle
        self.wait_time = wait_time

    def get_image_from_camera(self):
        time.sleep(self.wait_time)
        self.puzzle.process_events()
        # time.sleep(0.05)
        # self.puzzle.process_events()
        return self.puzzle["Camera"]["image"].value.astype(np.int16)
    
    # Should be useless as Camera piece can do it
    def set_backbround(self):
        self.background = self.get_image_from_camera()
        print(self.background.dtype)
    
    def get_processed_image(self):
        if not hasattr(self, "background"):
            self.background = 0
        return np.maximum(0, self.get_image_from_camera() - self.background)
    #

    def get_intensity(self):
        intensity = np.sum(self.get_processed_image())
        return intensity
    

def get_sorted_peak_idx(image, axis=1, threshold=100):
    arr = np.sum(image, axis=axis)
    local_max = (np.diff(arr, append=arr[-1]) < 0) & (np.diff(arr, prepend=arr[0]) > 0) & (arr > threshold)

    peak_idx = np.nonzero(local_max)[0]
    sorted_peaks_idx = np.argsort(arr[peak_idx])[::-1]

    return peak_idx[sorted_peaks_idx]
    
# n: order of polynomial
def fit(x, y, n):
    n += 1
    power = np.arange(n)
    X = np.tile(x, (n, 1)).T ** power
    P = np.linalg.inv(X.T @ X)
    parameters = P @ X.T @ y

    N, p = X.shape
    delta_sq = np.sum( (y - X @ parameters)**2 ) / (N-p)
    return parameters, np.diag(delta_sq * P)

def simulate_fit(x, params):
    n = params.size
    return np.tile(x, (n, 1)).T ** np.arange(0,n) @ params

def save_csv(x, y, name):
    data = np.vstack((x, y)).T
    np.savetxt(name, data)


def extract_one_cos2_period(grayscale, intensities, ignore_from_beginning=1):
    x = grayscale
    y = intensities
    slopes = np.sign(np.diff(y, append=y[-1]))
    
    exclude_first = np.ones_like(y, dtype=bool)
    exclude_first[:ignore_from_beginning+1] = 0    
    intensity_diff = np.abs(y - y[0])
    same_slope = (slopes == slopes[0]) & exclude_first
    intensity_diff_masked = intensity_diff[same_slope]
    idx_masked = np.arange(y.size)[same_slope]

    end_idx = idx_masked[np.argmin(intensity_diff_masked)]
    return x[0:end_idx], y[0:end_idx]
    

def map_grayscale_to_phase(grayscales, intensities, ignored_samples=1, ignored_from_beginning=1):
    # Preprocess input to extract one period and shift so that index 0 is maximum
    x, y = extract_one_cos2_period(grayscales, intensities, ignored_from_beginning)
    slopes = np.sign(np.diff(y, append=y[-1]+ (y[-1]-y[-2])))
    cos = np.sqrt(y) * slopes * -1
    raw_phase = 2*np.acos(cos)

    min_idx = np.argmin(y)
    ignore_mask = np.ones_like(raw_phase, dtype=bool)
    ignore_mask[min_idx-ignored_samples : min_idx+ignored_samples+1] = False

    return x[ignore_mask], raw_phase[ignore_mask]


def linear_interpolation(target, x, y, maximum=1023):
    """
    Given two arrays that (discretely) represent some function, apply that function to a given array 
    using linear interpolation.
    
    :param target: Array that you want to apply the function to.
    :param x: Arguments
    :param y: Images correspondting to the arguments
    :param maximum: Maximum value allowed in the returned array. Values above this parameter are clamped to its value.
    """
    if np.min(target) < np.min(x):
        raise ValueError("Linear interpolation failed: Some target values not in range of x")

    # Find what coefficients to use
    diff = np.tile(x, (target.size, 1)).T - target
    # interp_coef_idx = np.argmax(diff > 0, axis=0) - 1
    interp_coef_idx = x.size - np.argmax(diff[::-1, :] <= 0, axis=0) - 1
    interp_coef_idx[target > np.max(x)] -= 1  # target x bigger than np.max(s) use last interpolation slope
    # Repeat last value to avoid indexing issues
    x = np.append(x, x[-1] + 1)  # Avoid divison by zero
    y = np.append(y, y[-1])
    
    # Interpolation formula
    slope = (y[interp_coef_idx + 1] - y[interp_coef_idx]) / (x[interp_coef_idx + 1] - x[interp_coef_idx])
    result = y[interp_coef_idx] + (target - x[interp_coef_idx])*slope
    result[result > maximum] = maximum  # Clamp
    return result

def build_phase_to_grayscale_interpolator(phases, grayscales, period=2*np.pi):
    """
    Build a phase->grayscale linear interpolator that handles 2π wrapping.

    Parameters
    ----------
    phases : array_like
        Measured phases in radians (wrapped to [0, 2π) or close).
    grayscales : array_like
        Corresponding grayscale values (same length).
    period : float
        Phase period (default 2π).

    Returns
    -------
    f : callable
        f(phi) returns interpolated grayscale for phase(s) phi (radians).
    """
    
    # First remove the 2pi wrap
    phases = (phases - phases[0]) % period + phases[0]

    def f(phi):
        phi[phi < phases[0]] += period
        return linear_interpolation(phi, phases, grayscales)
    return f