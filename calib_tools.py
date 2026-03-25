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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


class ParameterScanner(pzp.Piece):
    """
    Abstract piece to scan a given parameter.
    """

    PARAM_SCAN_PARAM = "Scan parameter"
    PARAM_SCAN_TARGET = "Scanned parameter"
    PARAM_SCAN_MIN = "Start value"
    PARAM_SCAN_MAX = "End value (inclusive)"
    PARAM_SCAN_NB = "Sample nb"
    PARAM_SCAN_REP = "Repeat scan parameter nb"
    PARAM_SCAN_INTERVAL = "Sampling interval (ms)"
    PARAM_PROGRESS = "progress"

    ACTION_SCAN = "Scan"

    def __init__(self, puzzle):
        super().__init__(puzzle)
        self.stop = False
        self.data_proc_f = lambda x: x
        self.x = None
        self.y = None

    def define_params(self):
        pzp.param.progress(self, self.PARAM_PROGRESS)(None)
        pzp.param.text(self, self.PARAM_SCAN_PARAM, "piece:param", visible=False)(None)
        pzp.param.text(self, self.PARAM_SCAN_TARGET, "piece:param", visible=False)(None)
        pzp.param.spinbox(self, self.PARAM_SCAN_MIN, 0.0, visible=False)(None)
        pzp.param.spinbox(self, self.PARAM_SCAN_MAX, 1.0, visible=False)(None)
        pzp.param.spinbox(self, self.PARAM_SCAN_NB, 30, v_min=1)(None)
        pzp.param.spinbox(self, self.PARAM_SCAN_REP, 1, 1, 1000000)(None)
        pzp.param.spinbox(self, self.PARAM_SCAN_INTERVAL, 200.0, 1)(None)

    def define_actions(self):
        @pzp.action.define(self, self.ACTION_SCAN,)
        def scan():
            scan_param = pzp.parse.parse_params(self[self.PARAM_SCAN_PARAM].value, self.puzzle)[0]
            scanned_param = pzp.parse.parse_params(self[self.PARAM_SCAN_TARGET].value, self.puzzle)[0]
            scan_min, scan_max = self[self.PARAM_SCAN_MIN].value, self[self.PARAM_SCAN_MAX].value
            scan_nb, scan_int = self[self.PARAM_SCAN_NB].value, self[self.PARAM_SCAN_INTERVAL].value
            rep_nb = self[self.PARAM_SCAN_REP].value
            progress = self[self.PARAM_PROGRESS]

            dummy = scanned_param.value
            if type(dummy) is np.ndarray:
                init_value = np.zeros_like(dummy)
            else:
                init_value = 0

            scan_values = []
            scan_data = []
            self.stop = False
            for i in progress.iter(range(0, scan_nb+1)):
                value = scan_min + i*(scan_max - scan_min)/scan_nb
                scan_param.set_value(value)
                self.trigger_scan()

                sum_ = init_value.copy()
                for _ in range(rep_nb):
                    time.sleep(scan_int/1000)
                    sum_ += scanned_param.value
                    self.puzzle.process_events()

                data = sum_/rep_nb
                scan_values.append(value)
                scan_data.append(data)
                if self.stop:
                    break
            self.x = np.array(scan_values)
            self.y = np.array(scan_data) 
    
    # Function executed before taking a sample (eg use to send phase pattern to SLM)
    def trigger_scan(self):
        pass
        

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
            self["background_spec"].set_value(self["values"].get_value())
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
class BinaryCalib(ParameterScanner):
    """
    Piece that measures wavelength wise calibration data of the SLM. 2 modes are available; Grayscale calibration mode and Binary grating efficiency mode. 
    Uses OceanOptics spectrometer to fetch data.
    
    # Uniform pattern mode should be used with a 45deg polarizer (with respect to SLM operating axis) before and after SLM.

    Binary grating mode works only with a lens (at focal distance).
    
    :var Assumptions: Description
    :var arrays: Description
    """
    PARAM_NORMALIZE = "Normalize over"
    PARAM_FILENAME = "Save file name"
    PARAM_CALIB_DATA = "Calibration data"
    PARAM_CALIB_WL = "Calibration wl"
    PARAM_USE_CORRECTOR = "Apply phase correction"

    PARAM_SPEC_NAME = "Spectrometer piece name"
    PARAM_BINARY_NAME = "Binary grating pattern piece name"
    PARAM_CORRECTOR_NAME = "Pattern corrector piece name"
    PARAM_CALIB_BINARY_NAME = "Calibrated binary grating piece name"

    NORM_NONE = "None"
    NORM_ALL = "All"
    NORM_PER_WL = "Per wavelength"

    ACTION_MEASURE = "Measure"

    def define_params(self):
        pzp.param.dropdown(self, self.PARAM_NORMALIZE, self.NORM_NONE)([self.NORM_NONE, self.NORM_ALL, self.NORM_PER_WL])
        pzp.param.checkbox(self, self.PARAM_USE_CORRECTOR, False)(None)
        pzp.param.text(self, self.PARAM_FILENAME, f"binarycalib.csv")(None)
        pzp.param.array(self, self.PARAM_CALIB_DATA, False)(None)
        pzp.param.array(self, self.PARAM_CALIB_WL, False)(None)

        pzp.param.text(self, self.PARAM_SPEC_NAME, OceanSpectrometer.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_BINARY_NAME, pat.BinaryGratingPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_CORRECTOR_NAME, PhaseCorrector.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_CALIB_BINARY_NAME, pat.CalBinaryGratingPattern.__name__, visible=False)(None)
        
        super().define_params()
        pzp.action.settings(self)

        self[self.PARAM_SCAN_PARAM].set_value(f"{pat.BinaryGratingPattern.__name__}:{pat.BinaryGratingPattern.PARAM_PHASE}")
        self[self.PARAM_SCAN_TARGET].set_value(f"{OceanSpectrometer.__name__}:values")
        self[self.PARAM_SCAN_MAX].set_value(1023)
        self[self.PARAM_SCAN_MIN].set_value(0)

    def define_actions(self):
        super().define_actions()

        @pzp.action.define(self, "Plot and save")
        def plot():
            contrasts = self.x
            spectrums = self.y
            nm_per_wl = False
            nm_all = False
            name = self[self.PARAM_FILENAME].value
            wls = self.puzzle[self[self.PARAM_SPEC_NAME].value]["wls"].value
            if self[self.PARAM_NORMALIZE].value == self.NORM_PER_WL:
                nm_per_wl = True
                spectrums /= np.max(spectrums, axis=0)
            elif self[self.PARAM_NORMALIZE].value == self.NORM_ALL:
                nm_all = True
                spectrums /= np.max(spectrums)

            plt.imshow(spectrums, origin="lower", aspect="auto", extent=[np.min(wls), np.max(wls), 0, 1023])
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Phase (Grayscale)")
            cbar = plt.colorbar()
            cbar.set_label(f"{"Relative" if nm_all or nm_per_wl else ""} Intensity {"per wavelength" if nm_per_wl else ""}")
            plt.show()

            df = pd.DataFrame(spectrums, index=contrasts, columns=wls)
            df.to_csv(f"{name}")
    
    def trigger_scan(self):
        grating: pat.BinaryGratingPattern = self.puzzle[self[self.PARAM_BINARY_NAME].value]
        corrector = self.puzzle[self[self.PARAM_CORRECTOR_NAME].value]
        grating.actions[grating.ACTION_SEND]()
        if self[self.PARAM_USE_CORRECTOR].value:
            corrector.actions[corrector.ACTION_SEND]()
            

class PhaseCorrector(pat.PatternGenerator):
    """
    Piece that corrects binary grating pattern based on calibration data from UniformAndBinaryCalib
    """
    PARAM_MIN_WL = "Min. wavelength considered"
    PARAM_MAX_WL = "Max. wavelength considered"
    PARAM_WL_FILE = "Wavelength scan file name"
    PARAM_CALIB_FILE = "Calibration file name"
    PARAM_GRAYSCALES = "Grayscales"
    PARAM_MODE = "Method"
    # PARAM_INVERT_CORRECTION_ORDER = "Invert correction order"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_MIN_WL, 1179, 1, 9999999)(None)
        pzp.param.spinbox(self, self.PARAM_MAX_WL, 1378, 1, 9999999)(None)
        # pzp.param.checkbox(self, self.PARAM_INVERT_CORRECTION_ORDER, False)(None)
        pzp.param.text(self, self.PARAM_WL_FILE, "wl_scan.csv")(None)
        # Column by column (per wavelength) correction data. Each index correcspond to a (2,N) shape matrix that contains calibration curve.
        pzp.param.text(self, self.PARAM_CALIB_FILE, "binarycalib.csv", visible=True)(None)
        pzp.param.dropdown(self, self.PARAM_MODE, "Fit")(["Fit", "Interpolation"])
        super().define_params()

    def fitted_intensity(self, contrasts, fit_contrasts, fit_intensities):
        def amplitude(x, x0, k):
            # y = np.cos(x0*(x**2)+k*x)**2
            y = x0*(x**2)+k
            return y
        cos2_model = lambda x,A,B,C,E,F,G,H: A*amplitude(x/1024, E, F)*(np.cos(H*(x**0.7)+G*(x**1.2)+B*x + C)**2)

        A1 = np.max(fit_intensities) - np.min(fit_intensities)
        B1 = np.pi/800      # rough guess for frequency
        C1 = 0
        D1 = 0
        # E1 = 0
        E1 = -1
        # F1 = 0.4
        F1 = 1
        G1 = 0
        H1 = 0
        p1 = [A1,B1,C1,E1,F1,G1,H1]
        popt, pcov = curve_fit(cos2_model, fit_contrasts, fit_intensities, p0=p1, maxfev=10000)
        
        return cos2_model(contrasts, *popt)

    def fit_method(self, grayscales, intensities1, intensities2, wl, wl1, wl2):
        test_contrasts = np.linspace(0, 1023, 1200)
        test_intensity1 = self.fitted_intensity(test_contrasts, grayscales, intensities1)
        test_intensity2 = self.fitted_intensity(test_contrasts, grayscales, intensities2)
        
        int_interp = np.linspace(test_intensity1, test_intensity2)
        wl_interp = np.linspace(wl1, wl2)
        closest_wl_idx = np.argmin(np.abs(wl_interp - wl))
        
        test_intensity = int_interp[closest_wl_idx, :]

        min_idx = np.argmin(test_intensity)
        max1_idx = np.argmax(test_intensity[:min_idx])
        max2_idx = np.argmax(test_intensity[min_idx:]) + min_idx + 1

        first_half = test_intensity[max1_idx:min_idx]
        first_half /= np.max(first_half)
        first_inversed = np.acos(np.sqrt(first_half))
        
        second_half = test_intensity[min_idx: max2_idx]
        second_half /= np.max(second_half)
        second_inversed = np.pi - np.acos(np.sqrt(second_half))

        interp_contrasts = test_contrasts[max1_idx:max2_idx]
        contrast_to_phase = np.concatenate((first_inversed, second_inversed))
        
        def f(contrasts,x=interp_contrasts, y=contrast_to_phase):
            return np.interp(np.pi*contrasts/1024, y, x)
        return f
    
    def interp_method(self, grayscales, intensities):
        # 1. Smooth the data to eliminate noise (window_length must be odd)
        # May need to tweak window_length (e.g., 51, 101) based on sampling density
        smoothed = savgol_filter(intensities, window_length=6, polyorder=3)
        
        # 2. Find the global minimum (1 * pi phase)
        min_idx = np.argmin(smoothed)
        g_pi = grayscales[min_idx]
        I_min = smoothed[min_idx]
        
        # 3. Find the first local maximum (0 * pi phase) in the first half
        max1_idx = np.argmax(smoothed[:min_idx])
        g_0 = grayscales[max1_idx]
        I_max1 = smoothed[max1_idx]
        
        # 4. Find the second local maximum (2 * pi phase) in the second half
        max2_idx = min_idx + np.argmax(smoothed[min_idx:])
        g_2pi = grayscales[max2_idx]
        I_max2 = smoothed[max2_idx]
        
        # 5. Extract actual phase response from the smoothed curve 
        # Using I = I_min + (I_max - I_min)/2 * (1 + cos(phi)) -> phi = arccos(2 * I_norm - 1)
        active_gray = grayscales[max1_idx:max2_idx+1]
        active_I = smoothed[max1_idx:max2_idx+1]
        
        half1_mask = active_gray <= g_pi
        half2_mask = active_gray > g_pi
        
        phase_response = np.zeros_like(active_gray, dtype=float)
        
        # Calculate phase for 0 to pi
        I_norm1 = (active_I[half1_mask] - I_min) / (I_max1 - I_min)
        I_norm1 = np.clip(I_norm1, 0, 1) # Clip to avoid domain errors in arccos due to slight noise
        phase_response[half1_mask] = np.arccos(2 * I_norm1 - 1)
        
        # Calculate phase for pi to 2pi
        I_norm2 = (active_I[half2_mask] - I_min) / (I_max2 - I_min)
        I_norm2 = np.clip(I_norm2, 0, 1)
        phase_response[half2_mask] = 2 * np.pi - np.arccos(2 * I_norm2 - 1)
        
        # 6. Create the correction function
        # This interpolator maps actual phase (0 to 2pi) back to the required grayscale
        inv_interp = interp1d(phase_response, active_gray, kind='cubic', bounds_error=False, fill_value=(g_0, g_2pi))
        
        # Create a closure to save the interpolator for this specific wavelength
        def make_correction_fct(interpolator):
            def f(target_contrasts):
                # Assuming incoming target_contrasts are scaled 0-1023 for 0-2pi
                target_phases = target_contrasts * (2 * np.pi / 1023)
                return interpolator(target_phases)
            return f
        
        return make_correction_fct(inv_interp)

    def define_actions(self):
        @pzp.action.define(self, "Get correction data")
        def get_calib_data():
            df = pd.read_csv(self[self.PARAM_CALIB_FILE].value, index_col=0)
            data = df.values  # Spectrums
            grayscales = df.index.to_numpy()
            calib_wls = df.columns.to_numpy(dtype=float)

            # Determine what wavelength correspond to what column
            wl_scan = np.loadtxt(self[self.PARAM_WL_FILE].value).T
            scan_col, scan_wls = wl_scan[0, :], wl_scan[1, :]

            # A1 = 1
            # B1 = 0
            # p1 = [A1,B1]
            # popt, pcov = curve_fit(lambda x,A,B:A*x+B, scan_col, scan_wls, p0=p1, maxfev=10000)
            
            col_nb = self.check_slm_status()[1]
            columns = np.arange(col_nb)
            # wls =  popt[0]*columns + popt[1]
            wls = np.interp(columns, scan_col, scan_wls)
            self.correction_fct = []  # Phase to grayscale

            # Perform fit 
            for i, wl in enumerate(wls):
                closest_wl_idces = np.argsort(np.abs(calib_wls-wl))
                closest_wl_idx = closest_wl_idces[0]
                next_closest_wl_idx = closest_wl_idces[1]
                intensities_1 = data[:, closest_wl_idx]
                intensities_2 = data[:, next_closest_wl_idx]

                if self[self.PARAM_MODE].value == "Fit":
                    self.correction_fct.append(self.fit_method(grayscales, intensities_1, intensities_2, wl, calib_wls[closest_wl_idx], calib_wls[next_closest_wl_idx]))

                else:
                    self.correction_fct.append(self.interp_method(grayscales, intensities_1))
            return

        @pzp.action.define(self, "Show correction function")
        def plot():
            slm_dim = self.check_slm_status()
            plot_data = []
            max_wl = self[self.PARAM_MAX_WL].value
            min_wl = self[self.PARAM_MIN_WL].value

            wl_scan = np.loadtxt(self[self.PARAM_WL_FILE].value).T
            scan_col, scan_wls = wl_scan[0, ::-1], wl_scan[1, ::-1]

            for wl in range(min_wl, max_wl, 2):
                col = int(np.interp(wl, scan_wls, scan_col))
                plot_data.append(self.correction_fct[col](np.linspace(0,1023)))
            
            plt.figure()
            plt.imshow(np.array(plot_data).T, origin="lower", aspect="auto", extent=[min_wl, max_wl, 0, 2*np.pi]) 
            plt.xlabel("SLM column")
            plt.ylabel("Target phase")
            cbar = plt.colorbar()
            cbar.set_label(f"Calibrated grayscale")
            plt.title("Calibration function plot")
            plt.show()
        super().define_actions()

    def generate_pattern(self, slm_dim):
        pattern = self.puzzle[self.get_slm_piece_name()][SLMPiece.PARAM_IMAGE].value
        new_pattern = []

        for col_i in range(pattern.shape[1]):
            new_pattern.append(self.correction_fct[col_i](pattern[:, col_i]))

        corrected_pattern = np.array(new_pattern)
        return corrected_pattern.T


class WavelengthScanner(pzp.Piece):
    PARAM_PAT_MULTIPLIER_NAME = "Pattern multiplier piece name"
    PARAM_BINARY_NAME = "Binary grating piece name"
    PARAM_SLIT_NAME = "Slit pattern piece name"
    PARAM_SPEC_NAME = "Spectrometer piece name"
    PARAM_INTERVAL = "Sampling interval (ms)"
    PARAM_SAMPLE_NUM = "Sample nb."
    PARAM_SAVE_NAME = "Save file name"
    PARAM_MAX_WL = "Max WL kept (nm)"
    PARAM_MIN_WL = "Min WL kept (nm)"

    def define_params(self):
        pzp.param.text(self, self.PARAM_PAT_MULTIPLIER_NAME, pat.PatternMultiplier.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_BINARY_NAME, pat.BinaryGratingPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_SLIT_NAME, pat.SlitPattern.__name__, visible=False)(None)
        pzp.param.text(self, self.PARAM_SPEC_NAME, OceanSpectrometer.__name__, visible=False)(None)
        pzp.param.spinbox(self, self.PARAM_INTERVAL, 200, 1, 99999)(None)
        pzp.param.spinbox(self, self.PARAM_SAMPLE_NUM, 30, 1, 99999)(None)
        pzp.param.spinbox(self, self.PARAM_MAX_WL, 99999, 1, 9999999)(None)
        pzp.param.spinbox(self, self.PARAM_MIN_WL, 0, 1, 9999999)(None)
        pzp.param.text(self, self.PARAM_SAVE_NAME, "wl_scan.csv")(None)
        pzp.action.settings(self)
    
    def define_actions(self):
        @pzp.action.define(self, "Scan")
        def aaa():
            slit_pattern: pat.SlitPattern = self.puzzle[self[self.PARAM_SLIT_NAME].value]
            multiplier: pat.PatternMultiplier = self.puzzle[self[self.PARAM_PAT_MULTIPLIER_NAME].value]
            multiplier[multiplier.PARAM_GEN1].set_value(self[self.PARAM_BINARY_NAME].value)
            multiplier[multiplier.PARAM_GEN2].set_value(self[self.PARAM_SLIT_NAME].value)
            spec: OceanSpectrometer = self.puzzle[self[self.PARAM_SPEC_NAME].value]
            slm_dim = multiplier.check_slm_status()
            vertical_slit = slit_pattern[slit_pattern.PARAM_VERTICAL].value
            slm_length = slm_dim[1] if vertical_slit else slm_dim[0]
            interval_ms = self[self.PARAM_INTERVAL].value
            max_wl = self[self.PARAM_MAX_WL].value
            min_wl = self[self.PARAM_MIN_WL].value
            sample_nb= self[self.PARAM_SAMPLE_NUM].value
            
            data = []
            col_indices = []
            raw_data = []

            for i in range(1, sample_nb):
                offset_from_left_edge = (i/sample_nb)*slm_length
                slit_offset = int(-slm_length/2 + offset_from_left_edge)
                slit_pattern[slit_pattern.PARAM_OFFSET].set_value(slit_offset)
                multiplier.actions[multiplier.ACTION_SEND]()
                time.sleep(interval_ms / 1000)
                self.puzzle.process_events()
                spectrum = spec["values"].value
                wls = spec["wls"].value
                mask = (min_wl <= wls) & (wls <= max_wl)
                wls, spectrum = wls[mask], spectrum[mask]

                x_peak, popt, x_local, y_local = self.fit_gaussian_peak(wls, spectrum, n_points=7)
                data.append(x_peak)
                raw_data.append(spectrum)
                col_indices.append(slit_offset + slm_length/2)
            util.save_csv(np.array(col_indices), np.array(data), self[self.PARAM_SAVE_NAME].value)        
            df = pd.DataFrame(np.array(raw_data), index=np.array(col_indices), columns=wls)
            df.to_csv(f"raw_{self[self.PARAM_SAVE_NAME].value}")

            plt.figure()
            plt.plot(col_indices, data, 'k.-')
            plt.xlabel("Column index")
            plt.ylabel("Wavelength")
            plt.title("Column vs Wavelength")
            plt.show()

    def gaussian(self, x, A, x0, sigma, C):
        return A * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + C

    def fit_gaussian_peak(self, x, y, n_points=7):
        """
        Fit a Gaussian around the maximum using a small number of nearby points
        and return the fitted peak position.

        Parameters
        ----------
        x : array-like
            x values
        y : array-like
            y values
        n_points : int
            Number of points around the maximum to use for the fit

        Returns
        -------
        x_peak : float
            Fitted peak position
        popt : tuple
            Fitted Gaussian parameters (A, x0, sigma, C)
        x_fit : ndarray
            x values used for the local fit
        y_fit : ndarray
            y values used for the local fit
        """
        x = np.asarray(x)
        y = np.asarray(y)

        i_max = np.argmax(y)
        half = n_points // 2

        i0 = max(0, i_max - half)
        i1 = min(len(x), i_max + half + 1)

        # adjust window if near boundary
        if i1 - i0 < n_points:
            if i0 == 0:
                i1 = min(len(x), n_points)
            elif i1 == len(x):
                i0 = max(0, len(x) - n_points)

        x_fit = x[i0:i1]
        y_fit = y[i0:i1]

        # initial guesses
        A0 = y_fit.max() - y_fit.min()
        x00 = x[i_max]
        dx = np.mean(np.diff(x_fit)) if len(x_fit) > 1 else 1.0
        sigma0 = max(dx * len(x_fit) / 4, 1e-12)
        C0 = y_fit.min()

        p0 = [A0, x00, sigma0, C0]

        # optional bounds to keep fit reasonable
        bounds = (
            [0, x_fit.min(), 1e-12, -np.inf],
            [np.inf, x_fit.max(), np.inf, np.inf]
        )

        popt, _ = curve_fit(self.gaussian, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
        x_peak = popt[1]

        return x_peak, popt, x_fit, y_fit


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
    