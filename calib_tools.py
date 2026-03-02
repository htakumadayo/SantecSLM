import puzzlepiece as pzp
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from SantecSLM.pzp import SLMPiece
import SantecSLM.patterns as pat
# from pzp_hardware.oceanoptics import spectrometer
import SantecSLM.utility as util
import pandas as pd
from pyqtgraph.Qt import QtWidgets
from puzzlepiece.extras import hardware_tools as pht
import pyqtgraph as pg


class OceanSpectrometer(pzp.Piece):
    """
    A very basic Piece for getting values and wavelengths from an OceanOptics
    spectrometer. Contributions welcome to expose more options!

    .. image:: ../images/pzp_hardware.oceanoptics.spectrometer.Piece.png
    """
    custom_horizontal = True

    def define_params(self):
        @pzp.param.dropdown(self, "spectrometer", "")
        def list_spectrometers():
            if not self.puzzle.debug:
                return self.imports.list_devices()

        @pzp.param.connect(self)
        def connect():
            if self.puzzle.debug:
                return 1
            self.spec = self.imports.Spectrometer.from_serial_number(
                self.params['spectrometer'].get_value().split(":")[1][:-1]
            )

        @pzp.param.disconnect(self)
        def disconnect():
            if self.puzzle.debug:
                return 0
            if self._ensure(capture_exception=True):
                self.spec.close()
            return 0

        pzp.param.array(self, 'wls', False)(None)
        pzp.param.array(self, 'background_spec', False)(None)
        pzp.param.checkbox(self, 'Subtract background', True, True)(None)

        @pzp.param.array(self, 'values')
        @self._ensure
        def values():
            if self.puzzle.debug:
                wls, vals = np.arange(100), np.random.random(100)
                self.params['wls'].set_value(wls)
                return vals
            
            wls, vals = self.spec.spectrum()
            if self['Subtract background'].value and self['background_spec'].value is not None:
                vals = np.maximum(0, vals - self['background_spec'].value)
            self.params['wls'].set_value(wls)
            return vals
        
        @pzp.param.spinbox(self, "Integration time (μs)", 10000)
        @self._ensure
        def exposure(value):
            if self.puzzle.debug:
                return value
            self.spec.integration_time_micros(value)

    def define_actions(self):
        @pzp.action.define(self, "Set background")
        def set_background():
            self["background_spec"].set_value(self["values"].value)
            print("Background set.")

    @pzp.piece.ensurer
    def _ensure(self):
        if not self.puzzle.debug and not hasattr(self, 'spec'):
            raise("Spectrometer not connected")

    def custom_layout(self):
        layout = QtWidgets.QVBoxLayout()

        # The thread runs self.get_value repeatedly, which updates the plot through the
        # Signal connection defined below
        self.timer = pzp.threads.PuzzleTimer('Live', self.puzzle, self.params['values'].get_value, 0.05)
        layout.addWidget(self.timer)

        self.pw = pg.PlotWidget()
        layout.addWidget(self.pw)
        self.plot = self.pw.getPlotItem()
        self.plot_line = self.plot.plot([0], [0], symbol='o', symbolSize=3)

        # Update the plot when the values change (through a CallLater, so the
        # update is done only when the GUI loop is running)
        def update_plot():
            self.plot_line.setData(
                self.params['wls'].value,
                self.params['values'].value
            )
        update_later = pzp.threads.CallLater(update_plot)
        self.params['values'].changed.connect(update_later)

        return layout

    def setup(self):
        pht.requirements(
            {
                "seabreeze": {
                    "pip": "seabreeze",
                    "url": "https://python-seabreeze.readthedocs.io/en/latest/install.html"
                }
            }
        )
        import seabreeze.spectrometers
        self.imports = seabreeze.spectrometers
    

# Calibration 2: Uniform pattern, Grayscale calibration

# TOdo? maybe also check what happens if we add offset to grating
# Calibration 3: Binary grating efficiency.
class UniformAndBinaryCalib(pzp.Piece):
    """
    Piece that measures wavelength wise calibration data of the SLM. 2 modes are available; Grayscale calibration mode and Binary grating efficiency mode. 
    Uses OceanOptics spectrometer to fetch data.
    
    Uniform pattern mode should be used with a 45deg polarizer (with respect to SLM operating axis) before and after SLM.

    Binary grating mode works only with a lens (at focal distance).
    
    :var Assumptions: Description
    :var arrays: Description
    """
    PARAM_MODE = "Calibration mode"
    PARAM_SAMPLE_NB = "Sample number"
    PARAM_CAPT_INTERVAL = "Sampling interval (ms)"
    PARAM_MAX_WL = "Max wavelength (nm)"
    PARAM_MIN_WL = "Min wavelength (nm)"
    PARAM_NORMALIZE = "Normalize over"
    PARAM_FILENAME = "Save file name"
    PARAM_CALIB_DATA = "Calibration data"
    PARAM_CALIB_WL = "Calibration wl"
    PARAM_USE_CORRECTOR = "Apply phase correction"

    PARAM_SPEC_NAME = "Spectrometer piece name"
    PARAM_UNIFORM_NAME = "Uniform pattern piece name"
    PARAM_BINARY_NAME = "Binary grating pattern piece name"
    PARAM_CORRECTOR_NAME = "Pattern corrector piece name"

    MODE_EFF = "BinaryEfficiency"
    MODE_GRAY = "GrayscaleCalibration"

    NORM_NONE = "None"
    NORM_ALL = "All"
    NORM_PER_WL = "Per wavelength"

    ACTION_MEASURE = "Measure"

    def define_params(self):
        pzp.param.dropdown(self, self.PARAM_MODE, self.MODE_EFF)([self.MODE_EFF, self.MODE_GRAY])
        pzp.param.spinbox(self, self.PARAM_SAMPLE_NB, 30, 1, 9999)(None)
        pzp.param.spinbox(self, self.PARAM_CAPT_INTERVAL, 50.0, 0.05, 9999, v_step=5.0)(None)
        pzp.param.spinbox(self, self.PARAM_MIN_WL, 1050, 1, 99999)(None)
        pzp.param.spinbox(self, self.PARAM_MAX_WL, 1550, 1, 99999)(None)
        pzp.param.dropdown(self, self.PARAM_NORMALIZE, self.NORM_NONE)([self.NORM_NONE, self.NORM_ALL, self.NORM_PER_WL])
        pzp.param.checkbox(self, self.PARAM_USE_CORRECTOR, False)(None)
        pzp.param.text(self, self.PARAM_FILENAME, f"calib{self.MODE_EFF}.csv")(None)
        pzp.param.array(self, self.PARAM_CALIB_DATA, False)(None)
        pzp.param.array(self, self.PARAM_CALIB_WL, False)(None)

        pzp.param.text(self, self.PARAM_SPEC_NAME, OceanSpectrometer.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_UNIFORM_NAME, pat.UniformPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_BINARY_NAME, pat.BinaryGratingPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_CORRECTOR_NAME, PhaseCorrector.__name__, visible=False)(None)
        pzp.action.settings(self)

    def define_actions(self):
        @pzp.action.define(self, self.ACTION_MEASURE)
        def measure():
            sample_nb = self[self.PARAM_SAMPLE_NB].value
            min_wl, max_wl = self[self.PARAM_MIN_WL].value, self[self.PARAM_MAX_WL].value
            wait_time = self[self.PARAM_CAPT_INTERVAL].value / 1000
            grating: pat.BinaryGratingPattern = self.puzzle[self[self.PARAM_BINARY_NAME].value]
            uniform: pat.UniformPattern = self.puzzle[self[self.PARAM_UNIFORM_NAME].value]
            spec: OceanSpectrometer = self.puzzle[self[self.PARAM_SPEC_NAME].value]
            save_name = self[self.PARAM_FILENAME].value

            spec_wavelengths = spec["wls"].value
            spec_mask = (min_wl <= spec_wavelengths) & (spec_wavelengths <= max_wl)

            mode = self[self.PARAM_MODE].value
            contrasts = np.linspace(0, 1023, sample_nb).astype(int)
            spectrums = [None] * sample_nb

            if mode == self.MODE_EFF:
                grating[pat.BinaryGratingPattern.PARAM_DUTY_CYCLE].set_value(0.5)

            target_piece = grating if self.MODE_EFF == mode else uniform
            target_piece[target_piece.PARAM_PHASE].set_value(0)
            target_piece.actions[target_piece.ACTION_SEND]()
            time.sleep(1)
            self.puzzle.process_events()

            for i, contrast in enumerate(contrasts):
                target_piece[target_piece.PARAM_PHASE].set_value(contrast)
                target_piece.actions[target_piece.ACTION_SEND]()
                if self[self.PARAM_USE_CORRECTOR].value:
                    self.puzzle[self[self.PARAM_CORRECTOR_NAME].value].actions[pat.PatternGenerator.ACTION_SEND]()
                time.sleep(wait_time)
                self.puzzle.process_events()
                spectrums[i] = spec["values"].value[spec_mask]

            spectrums = np.array(spectrums)

            nm_per_wl = False
            nm_all = False
            if self[self.PARAM_NORMALIZE].value == self.NORM_PER_WL:
                nm_per_wl = True
                spectrums /= np.max(spectrums, axis=0)
            elif self[self.PARAM_NORMALIZE].value == self.NORM_ALL:
                nm_all = True
                spectrums /= np.max(spectrums)
            effective_wls = spec_wavelengths[spec_mask]

            self[self.PARAM_CALIB_DATA].set_value(spectrums)
            self[self.PARAM_CALIB_WL].set_value(effective_wls)

            plt.imshow(spectrums, origin="lower", aspect="auto", extent=[np.min(effective_wls), np.max(effective_wls), 0, 1023])
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Phase (Grayscale)")
            cbar = plt.colorbar()
            cbar.set_label(f"{"Relative" if nm_all or nm_per_wl else ""} Intensity {"per wavelength" if nm_per_wl else ""}")
            plt.savefig(f"{save_name}.svg")
            plt.show()

            df = pd.DataFrame(spectrums, index=contrasts, columns=effective_wls)
            df.to_csv(f"{save_name}.csv")


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



class PhaseCorrector(pat.PatternGenerator):
    """
    Piece that corrects binary grating pattern based on calibration data from UniformAndBinaryCalib
    """
    PARAM_MIN_WL = "Min. wavelength"
    PARAM_MAX_WL = "Max. wavelength"
    PARAM_IGNORE = "Ignored samples around min."
    PARAM_IGNORE_B = "Ignored samples at beginning"
    PARAM_CORRECTION = "Correction data"
    PARAM_CALIB_FILE = "Calibration file name"
    PARAM_GRAYSCALES = "Grayscales"
    PARAM_INVERT_CORRECTION_ORDER = "Invert correction order"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_MIN_WL, 1050, 1, 9999999)(None)
        pzp.param.spinbox(self, self.PARAM_MAX_WL, 1400, 1, 9999999)(None)
        pzp.param.spinbox(self, self.PARAM_IGNORE, 1, 0, 9999)(None)
        pzp.param.spinbox(self, self.PARAM_IGNORE_B, 1, 0, 9999)(None)
        pzp.param.checkbox(self, self.PARAM_INVERT_CORRECTION_ORDER, False)(None)
        # Column by column (per wavelength) correction data. Each index correcspond to a (2,N) shape matrix that contains calibration curve.
        pzp.param.array(self, self.PARAM_CORRECTION, False)(None) 
        pzp.param.array(self, self.PARAM_GRAYSCALES, False)(None)
        pzp.param.text(self, self.PARAM_CALIB_FILE, "a.csv", visible=True)(None)
        super().define_params()

    def define_actions(self):
        @pzp.action.define(self, "Get correction data")
        def get_calib_data():
            df = pd.read_csv(self[self.PARAM_CALIB_FILE].value, index_col=0)
            data = df.values  # Spectrums
            contrasts_loaded = df.index.to_numpy()
            wls = df.columns.to_numpy(dtype=float)

            self[self.PARAM_GRAYSCALES].set_value(contrasts_loaded)
            print(wls.shape, data.shape)
            colbycol_calib = []
            col_nb = self.puzzle[self.get_slm_piece_name()][SLMPiece.PARAM_IMAGE].value.shape[1]
            max_wl, min_wl = self[self.PARAM_MAX_WL].value, self[self.PARAM_MIN_WL].value
            for col in range(col_nb):   # Are wavelengths scattered linearly?? -> Assume yes
                assumed_wl = min_wl + col*(max_wl - min_wl)/col_nb
                calib_idx = np.argmin(np.abs(wls - assumed_wl))
                calib = data[:, calib_idx]
                colbycol_calib.append(calib)
            self[self.PARAM_CORRECTION].set_value(np.stack(colbycol_calib))

        @pzp.action.define(self, "Show correction function")
        def plot():
            correction_data = self[self.PARAM_CORRECTION].value
            grayscales = self[self.PARAM_GRAYSCALES].value
            max_wl, min_wl = self[self.PARAM_MAX_WL].value, self[self.PARAM_MIN_WL].value
            ignore = self[self.PARAM_IGNORE].value
            ignore_b = self[self.PARAM_IGNORE_B].value
            pattern = self.puzzle[self.get_slm_piece_name()][SLMPiece.PARAM_IMAGE].value.T * 2*np.pi / 1024
            wls = np.zeros(pattern.shape[0])
            plot_data = []            
            for col in range(pattern.shape[0]):
                wls[col] = min_wl + (max_wl - min_wl)*col/pattern.shape[0]
                grayscale, phase = map_grayscale_to_phase(grayscales, correction_data[col], ignore, ignore_b)
                f = build_phase_to_grayscale_interpolator(phase, grayscale)
                plot_data.append(f(np.linspace(0, 2*np.pi)))
            
            plt.figure()
            plt.imshow(np.array(plot_data), origin="lower", aspect="auto", extent=[min_wl, max_wl, 0, 1023]) 
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Original grayscale")
            plt.title("Calibration function plot")
            plt.show()


        super().define_actions()

    def generate_pattern(self, slm_dim):
        pattern = self.puzzle[self.get_slm_piece_name()][SLMPiece.PARAM_IMAGE].value.T * 2*np.pi / 1024
        ignore = self[self.PARAM_IGNORE].value
        ignore_b = self[self.PARAM_IGNORE_B].value
        inverted = self[self.PARAM_INVERT_CORRECTION_ORDER].value
        correction_data = self[self.PARAM_CORRECTION].value
        grayscales = self[self.PARAM_GRAYSCALES].value
        corrected_grayscales = []
        for row in range(pattern.shape[0]):  # I realized later that wavelengths are spread along axis 1 and not 0
            inverted_row = correction_data.shape[0] - row - 1
            grayscale, phase = map_grayscale_to_phase(grayscales, correction_data[row if not inverted else inverted_row,:], ignore, ignore_b)
            f = build_phase_to_grayscale_interpolator(phase, grayscale)

            # corrected_grayscale = linear_interpolation(pattern[row, :], phase, grayscale)
            corrected_grayscale = f(pattern[row, :])
            corrected_grayscales.append(corrected_grayscale)
        corrected_pattern = np.stack(corrected_grayscales).T.astype(int)  # So fix it here
        return corrected_pattern


class WavelengthScanner(pzp.Piece):
    PARAM_PAT_MULTIPLIER_NAME = "Pattern multiplier piece name"
    PARAM_BINARY_NAME = "Binary grating piece name"
    PARAM_SLIT_NAME = "Slit pattern piece name"
    PARAM_SPEC_NAME = "Spectrometer piece name"
    PARAM_INTERVAL = "Sampling interval (ms)"
    PARAM_SAMPLE_NUM = "Sample nb."
    PARAM_SAVE_NAME = "Save file name"

    def define_params(self):
        pzp.param.text(self, self.PARAM_PAT_MULTIPLIER_NAME, pat.PatternMultiplier.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_BINARY_NAME, pat.BinaryGratingPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_SLIT_NAME, pat.SlitPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_SPEC_NAME, OceanSpectrometer.__name__, visible=False)(None)
        pzp.param.spinbox(self, self.PARAM_INTERVAL, 200, 1, 99999)(None)
        pzp.param.spinbox(self, self.PARAM_SAMPLE_NUM, 30, 1, 99999)(None)
        pzp.action.settings()
    
    def define_actions(self):
        slit_pattern: pat.SlitPattern = self.puzzle[self.PARAM_SLIT_NAME]
        multiplier: pat.PatternMultiplier = self.puzzle[self[self.PARAM_PAT_MULTIPLIER_NAME].value]
        multiplier[multiplier.PARAM_GEN1].set_value(self[self.PARAM_BINARY_NAME].value)
        multiplier[multiplier.PARAM_GEN2].set_value(self[self.PARAM_SLIT_NAME].value)
        spec: OceanSpectrometer = self.puzzle[self[self.PARAM_SPEC_NAME].value]
        slm_dim = multiplier.check_slm_status()
        vertical_slit = slit_pattern[slit_pattern.PARAM_VERTICAL].value
        slm_length = slm_dim[1] if vertical_slit else slm_dim[0]
        interval_ms = self[self.PARAM_INTERVAL].value
        sample_nb= self[self.PARAM_SAMPLE_NUM].value
        data = np.zeros(sample_nb)
        col_indices = np.zeros(sample_nb)

        for i in range(sample_nb):
            slit_offset = int(-slm_length/2 + (i/sample_nb)*slm_length)
            slit_pattern[slit_pattern.PARAM_OFFSET] = slit_offset
            multiplier.actions[multiplier.ACTION_SEND]()
            time.sleep(interval_ms / 1000)
            spectrum = spec["values"].value
            wls = spec["wls"].value
            max_idx = np.argmax(spectrum)

            data[i] = wls[max_idx]
            col_indices[i] = int(slm_length * i/sample_nb)
        util.save_csv(col_indices, data, self[self.PARAM_SAVE_NAME].value)        

        plt.figure()
        plt.plot(col_indices, data, 'k.-')
        plt.xlabel("Column index")
        plt.ylabel("Wavelength")
        plt.title("Column vs Wavelength")
        plt.show()


class Polarizer45degHelper(pzp.Piece):
    """
    Helper piece to align polariser at 45deg with respect to SLM operating axis
    """

    PARAM_UNIFORM = "Uniform pattern piece name"
    PARAM_INTERVAL = "Sampling interval (ms)"
    PARAM_SPEC_NAME = "Spectrometer piece name"
    PARAM_FOCUS_WL = "Focused wavelength"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_INTERVAL, 50, 1, 5000, v_step=50)(None)
        pzp.param.spinbox(self, self.PARAM_FOCUS_WL, 1275, 1, 99999)(None)
        pzp.param.text(self, self.PARAM_UNIFORM, pat.UniformPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_SPEC_NAME, OceanSpectrometer.__name__, visible=False)(None)

    def define_actions(self):
        @pzp.action.define(self, "Find max and min locations")
        def find_max_min():
            sample_nb = 40
            self.uniform_generator = self.puzzle[self[self.PARAM_UNIFORM].value]
            self.spectrometer = self.puzzle[self[self.PARAM_SPEC_NAME].value]
            phases = np.linspace(0, 1023, sample_nb, dtype=int)
            intensities = np.zeros_like(phases)

            self.set_phase_and_get_intensity(0)
            time.sleep(1)

            for i, phase in enumerate(phases):
                intensity = self.set_phase_and_get_intensity(phase)
                intensities[i] = intensity
                self.puzzle.process_events()

            # Initial guess
            A0 = np.max(intensities) - np.min(intensities)
            E0 = np.min(intensities)
            C0 = np.pi/1023      # rough guess for frequency
            D0 = 0.0      # phase guess
            p0 = [A0, C0, D0, E0]

            # Fit
            cos2_model = lambda x,A,C,D,E: A*(np.cos(C*x+D)**2)+E
            popt, pcov = curve_fit(cos2_model, phases, intensities, p0=p0)

            # Find min and max locations
            C_fit = popt[1]
            D_fit = popt[2]

            test = np.linspace(0, 1023, 2000, dtype=int)
            test_image = cos2_model(test, *popt)
            min_x = test[np.argmin(test_image)]
            max_x = test[np.argmax(test_image)]
            self.min_max = (min_x, max_x)

            # Just a preview
            x_fit = np.linspace(0, 1023, 1000)
            y_fit = cos2_model(x_fit, *popt)
            plt.scatter(phases, intensities)
            plt.plot(x_fit, y_fit)
            plt.scatter((min_x, max_x), cos2_model(np.array((min_x,max_x)), *popt), c='r')
            plt.show()
        
        @pzp.action.define(self, "Measure contrast")
        def measure_contrast():
            if not hasattr(self, "min_max"):
                raise RuntimeError("Please first measure min and max locations")
            
            min_intensity = self.set_phase_and_get_intensity(self.min_max[0])
            max_intensity = self.set_phase_and_get_intensity(self.min_max[1])
            print(f"Minimum: {min_intensity:.1f}, Maximum: {max_intensity:.1f}, Extinction ratio: {min_intensity/max_intensity:.4f}")


    def set_phase_and_get_intensity(self, phase):
        interval = self[self.PARAM_INTERVAL].value
        self.uniform_generator[pat.UniformPattern.PARAM_PHASE].set_value(phase)
        self.uniform_generator.actions[pat.PatternGenerator.ACTION_SEND]()
        time.sleep(interval/1000)
        spec = self.spectrometer["values"].value
        wls = self.spectrometer["wls"].value
        idx = np.argmin(np.abs(wls - self[self.PARAM_FOCUS_WL].value))
        print(wls[idx])
        return spec[idx]
    