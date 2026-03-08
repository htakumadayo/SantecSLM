from SantecSLM.pzp import SLMPiece
import SantecSLM.utility as util
import SantecSLM.patterns as pat
import puzzlepiece as pzp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# in meters
pixel_pitch = 8*1e-6
slm_name = SLMPiece.__name__

class PatternGenerator(pzp.Piece):
    """
    Base class of pattern generating puzzlepieces for SLM. Initially designed for Santec SLM, but can
    be easily adapted to other SLMs by modifying the methods (check_slm_status and send_image_to_slm).
    Concrete pattern generating pieces should inherit this class for simplicity and modularity.
    
    Most of common tasks (eg. sending the pattern to SLM, check error, etc..) of a generator are already defined in this class.
    Therefore simplest generators can be made by simply overriding the method generate_pattern(...) 

    Some generators have dependence on other pieces; they reference to them by default by class name. 

        ...

Methods
-------
define_params(self)
    Should be overridden (AND called by super().define_params() ) if one wishes to add custom parameters. 

define_actions(self)
    Should be overridden (AND called by super().define_actions() ) if one wishes to add custom actions. 

check_slm_status(self)
    

generate_pattern(self, slm_dim)
    This function is only called from method send_image_to_slm(...). Generates pattern that will be sent to SLM.
"""


    ACTION_SEND = "Generate pattern"
    PARAM_SLM_NAME = "SLM Piece name"

    def define_params(self):
        pzp.param.text(self, self.PARAM_SLM_NAME, slm_name, visible=False)(None)
        pzp.action.settings(self)

    def define_actions(self):
        @pzp.action.define(self, self.ACTION_SEND)
        def generate():
            self.send_image_to_slm()
    
    def get_slm_piece_name(self):
        """
        Alias of self[self.PARAM_SLM_NAME].value
        """
        return self[self.PARAM_SLM_NAME].value

    def check_slm_status(self):
        """
        Check SLM availability and dimension.
        
        Returns:
            If the SLM is connected, return dimension of slm, in the format (height, width)
            Otherwise return None.
        """
        slm = self.puzzle[slm_name]
        if slm[SLMPiece.PARAM_CONNECTED].value is not True:
            raise RuntimeError("SLM is not connected.")
        slm_dim = slm[SLMPiece.PARAM_SLM_DIMENSIONS].value
        return slm_dim
    
    def generate_pattern(self, slm_dim):
        """
        Generates pattern that will be sent to SLM. Internal method called only from send_image_to_slm().

        Override this method to make custom pattern generators.

        Args:
            slm_dim (tuple[int,int]): contains two integers (height, width), which are returned from check_slm_status()
        
        Returns:
            A numpy array of shape slm_dim. Any coefficient should be in the range [0, 1023].
        """
        return np.zeros(slm_dim)

    def send_image_to_slm(self):
        self.puzzle[slm_name][SLMPiece.PARAM_IMAGE].set_value(self.generate_pattern(self.check_slm_status()))
    

class UniformPattern(PatternGenerator):
    PARAM_PHASE = "Phase"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_PHASE, 50, 0, 1023)(None)
        super().define_params()
    
    def generate_pattern(self, slm_dim):
        if slm_dim is None:
            return
        phase = self[self.PARAM_PHASE].value
        pattern = np.ones(slm_dim)*phase
        return pattern
    

class BlazedGratingPattern(PatternGenerator):
    PARAM_PERIOD = "Period (px)"
    PARAM_PHASE = "Contrast"
    PARAM_HORIZONTAL = "Horizontal grating"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_PERIOD, 50, 2, 3000)(None)
        pzp.param.spinbox(self, self.PARAM_PHASE, 1023, 0, 1023)(None)
        pzp.param.checkbox(self, self.PARAM_HORIZONTAL, False)(None)
        super().define_params()

    def blazed_grating(self, height, width, period_px, max_phase):
        if self[self.PARAM_HORIZONTAL].value:
            x = np.arange(width)
            k = x % period_px
            row = k * max_phase / (period_px - 1)  # shape (width,)
            grating = np.tile(row, (height, 1))  # shape (height, width)
        else:
            y = np.arange(height)
            k = y % period_px
            col = k * max_phase / (period_px - 1)  # shape (width,)
            grating = np.tile(col, (width, 1)).T  # shape (height, width)
        return grating
    
    def generate_pattern(self, slm_dim):
        slm_dim = self.check_slm_status()
        period = self[self.PARAM_PERIOD].value
        max_phase = self[self.PARAM_PHASE].value
        pattern = self.blazed_grating(slm_dim[0], slm_dim[1], period, max_phase)
        return pattern


class BinaryGratingPattern(PatternGenerator):
    PARAM_PHASE = "Contrast"
    PARAM_PERIOD = "Period (px)"
    PARAM_DUTY_CYCLE = "Duty cycle"
    PARAM_HORIZONTAL = "Horizontal grating"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_PHASE, 512, 0, 1023)(None)
        pzp.param.spinbox(self, self.PARAM_PERIOD, 25, 2, 1023)(None)
        pzp.param.spinbox(self, self.PARAM_DUTY_CYCLE, 0.5, 0.0, 1.0, v_step=0.05)(None)
        pzp.param.checkbox(self, self.PARAM_HORIZONTAL, False)(None)
        super().define_params()
    
    def generate_pattern(self, slm_dim):
        slm_dim = self.check_slm_status()
        phase = self[self.PARAM_PHASE].value
        period = self[self.PARAM_PERIOD].value
        duty_cycle = self[self.PARAM_DUTY_CYCLE].value
        pattern = np.zeros(slm_dim)
        if self[self.PARAM_HORIZONTAL].value:
            pattern[:, np.arange(0, slm_dim[1])%period < period*duty_cycle] = phase
        else:
            pattern[np.arange(0, slm_dim[0])%period < period*duty_cycle, :] = phase
        return pattern


def clamp(value, min_, max_):
    return min(max(value, min_), max_)

class SlitPattern(PatternGenerator):
    PARAM_VERTICAL = "Vertical"
    PARAM_WIDTH = "Slit width (px)"
    PARAM_OFFSET = "Offset (px)"
    PARAM_DOUBLE =  "Double slits"
    PARAM_DISTANCE = "Slit separation (center to center) (px)"
    PARAM_SLIT_PHASE = "Slit phase"
    PARAM_NONSLIT_PHASE = "Non-slit phase"


    def define_params(self):
        pzp.param.checkbox(self, self.PARAM_VERTICAL, 1)(None)
        pzp.param.spinbox(self, self.PARAM_WIDTH, 50, 1, 1000, v_step=5)(None)
        pzp.param.spinbox(self, self.PARAM_OFFSET, 0.0, -1000, 1000, v_step=5)(None)

        pzp.param.checkbox(self, self.PARAM_DOUBLE, 0)(None)
        pzp.param.spinbox(self, self.PARAM_DISTANCE, 200, 1, 1000, v_step=5)(None)

        pzp.param.spinbox(self, self.PARAM_SLIT_PHASE, 0, 0, 1023)(None)
        pzp.param.spinbox(self, self.PARAM_NONSLIT_PHASE, 512, 0, 1023)(None)
        super().define_params()

    def draw_slit(self, pattern, vertical, width, offset, slm_dim, phase):
        half_width_px = width/2
        offset_px = offset
        slm_length_px = (slm_dim[1] if vertical else slm_dim[0])
        center_idx = round(slm_length_px / 2) + offset_px

        start = int(clamp(center_idx - half_width_px, 0, slm_length_px))
        end = int(clamp(center_idx + half_width_px, 0, slm_length_px))
        if vertical:
            pattern[:, start:end] = phase
        else:
            pattern[start:end, :] = phase

    def generate_pattern(self, slm_dim):
        slm_dim = self.check_slm_status()
        vertical = self[self.PARAM_VERTICAL].value
        double = self[self.PARAM_DOUBLE].value
        width = self[self.PARAM_WIDTH].value
        offset = self[self.PARAM_OFFSET].value
        slit_phase = self[self.PARAM_SLIT_PHASE].value
        nonslit_phase = self[self.PARAM_NONSLIT_PHASE].value

        pattern = np.ones(slm_dim)*nonslit_phase
        if double:
            distance = self[self.PARAM_DISTANCE].value
            self.draw_slit(pattern, vertical, width, offset + distance/2, slm_dim, slit_phase)
            self.draw_slit(pattern, vertical, width, offset - distance/2, slm_dim, slit_phase)
        else:
            self.draw_slit(pattern, vertical, width, offset, slm_dim, slit_phase)
        return pattern


class PinholePattern(PatternGenerator):
    PARAM_RADIUS = "Slit radius (px)"
    PARAM_OFFSET_X = "Offset X (px)"
    PARAM_OFFSET_Y = "Offset Y (px)"
    PARAM_SLIT_PHASE = "Slit phase"
    PARAM_NONSLIT_PHASE = "Non-slit phase"

    def define_params(self):
        pzp.param.spinbox(self, self.PARAM_RADIUS, 30, 1, 1000, v_step=5)(None)
        pzp.param.spinbox(self, self.PARAM_OFFSET_X, 0.0, -1000, 1000, v_step=5)(None)
        pzp.param.spinbox(self, self.PARAM_OFFSET_Y, 0.0, -1000, 1000, v_step=5)(None)

        pzp.param.spinbox(self, self.PARAM_SLIT_PHASE, 0, 0, 1023)(None)
        pzp.param.spinbox(self, self.PARAM_NONSLIT_PHASE, 512, 0, 1023)(None)
        super().define_params()

    def generate_pattern(self, slm_dim):
        slm_dim = self.check_slm_status()
        radius = self[self.PARAM_RADIUS].value
        offset_x = self[self.PARAM_OFFSET_X].value
        offset_y = self[self.PARAM_OFFSET_Y].value
        slit_phase = self[self.PARAM_SLIT_PHASE].value
        nonslit_phase = self[self.PARAM_NONSLIT_PHASE].value

        pattern = np.ones(slm_dim)*nonslit_phase
        yy, xx = np.mgrid[:slm_dim[0], :slm_dim[1]]
        mask = ((xx - offset_x - slm_dim[1]/2)**2 + (yy - offset_y - slm_dim[0]/2)**2) < (radius**2)
        pattern[mask] = slit_phase
        return pattern


class PatternMultiplier(PatternGenerator):
    """
    Blend two patterns by multiplying color values. 1023 corresponds to 1 and 0 to 0.
    Useful if one wants to combine multiple patterns. More than 2 patterns can be multiplied by nesting this piece.
    """

    PARAM_GEN1 = "Pattern generator 1"
    PARAM_GEN2 = "Pattern generator 2"

    def define_params(self):
        pzp.param.text(self, self.PARAM_GEN1, BinaryGratingPattern.__name__)(None)
        pzp.param.text(self, self.PARAM_GEN2, SlitPattern.__name__)(None)
        super().define_params()

    def generate_pattern(self, slm_dim):
        pattern1 = self.puzzle[self[self.PARAM_GEN1].value].generate_pattern(slm_dim)  / 1023.0
        pattern2 = self.puzzle[self[self.PARAM_GEN2].value].generate_pattern(slm_dim)  / 1023.0

        new_pattern = (pattern1 * pattern2 * 1023).astype(int)
        return new_pattern


class CalBinaryGratingPattern(PatternGenerator):
    PARAM_INT = "Spectral intensity formula"
    PARAM_PHASE = "Spectral phase formula"
    PARAM_PERIOD = "Period (px)"
    PARAM_CALIB_FILE = "Calibration file name"
    PARAM_WL_FILE = "Wavelength scan file name"

    def define_actions(self):
        @pzp.action.define(self, "Load calibration data")
        def a():
            df = pd.read_csv(self[self.PARAM_CALIB_FILE].value, index_col=0)
            data = df.values  # Spectrums
            grayscales = df.index.to_numpy()
            calib_wls = df.columns.to_numpy(dtype=float)

            # Determine what wavelength correspond to what column
            wl_scan = np.loadtxt(self[self.PARAM_WL_FILE].value).T
            scan_col, scan_wls = wl_scan[0, :], wl_scan[1, :]

            colbycol_calib = []
            col_nb = self.puzzle[self.get_slm_piece_name()][SLMPiece.PARAM_IMAGE].value.shape[1]
            columns = np.arange(col_nb)
            wls = util.linear_interpolation(columns, scan_col, scan_wls, 999999)
            calib_use_col = np.argmin(np.abs(np.tile(calib_wls, (wls.size, 1)).T - wls), axis=0)  # size=col_nb
            self.correction_fct = []

            # Perform fit 
            for col in calib_use_col: 
                fit_data = data[:, col]
                min_idx = np.argmin(fit_data) + int(0.15*fit_data.shape[0])
                fit_data = fit_data[:min_idx]
                fit_contrasts = grayscales[:min_idx]

                A0,B0,C0,D0 = np.max(fit_data) - np.min(fit_data), np.pi/800, 0, 0
                E0,F0,G0,H0 = A0/15, B0, 0, 0
                p0 = [A0, B0, C0, D0, E0, F0, G0, H0]
                cos2_model = lambda x,A,B,C,D,E,F,G,H: A*(np.cos(D*(x**2)+B*x + C)**2) + E*(np.cos(H*(x**2)+F*x + G)**4)
                popt, pcov = curve_fit(cos2_model, fit_contrasts, fit_data, p0=p0)

                x_new = np.arange(0, 1024)
                y_new = cos2_model(x_new, *popt)
                y_new = y_new[:np.argmin(y_new)]
                min_idx, max_idx = np.argmin(y_new), np.argmax(y_new)

                lookup_begin, lookup_end = min(min_idx, max_idx), max(min_idx, max_idx)
                lookup_x = x_new[lookup_begin:lookup_end]   
                lookup_y = y_new[lookup_begin:lookup_end]
                lookup_y_normalized = lookup_y/np.max(lookup_y)

                def interp(intensities, lookup_y_normalized=lookup_y_normalized, lookup_x=lookup_x):
                    return np.interp(intensities, lookup_y_normalized[::-1], lookup_x[::-1])  # Assumed max before min
                self.correction_fct.append(interp)
            return
        
        @pzp.action.define("Show calibration plot")
        def b():
            im = []
            intensities = np.linspace(0, 1)
            for fct in self.correction_fct:
                im.append(fct(intensities))

            plt.imshow(np.array(im).T, origin="lower", aspect="auto", extent=[0, len(self.correction_fct)-1, 0, 1])
            plt.xlabel("SLM column")
            plt.ylabel("Intensity")
            cbar = plt.colorbar()
            cbar.set_label(f"Contrast")
            plt.show()
        
        super().define_actions()

    def define_params(self):
        pzp.param.text(self, self.PARAM_INT, "1")(None)
        pzp.param.text(self, self.PARAM_PHASE, "0")(None)
        pzp.param.spinbox(self, self.PARAM_PERIOD, 25, 2, 1023)(None)
        pzp.param.text(self, self.PARAM_CALIB_FILE, "binaryefficiency.csv")(None)
        pzp.param.text(self, self.PARAM_WL_FILE, "wavelengthscan.csv")(None)
        super().define_params()

    def generate_pattern(self, slm_dim):
        slm_dim = self.check_slm_status()
        period = self[self.PARAM_PERIOD].value
        duty_cycle = 0.5
        pattern = np.zeros(slm_dim)
        target_attn = self[self.PARAM_ATTN].value

        for col in range(slm_dim[1]):
            pattern[np.arange(0, slm_dim[0])%period < period*duty_cycle, col] = self.correction_fct[col](target_attn)
        return pattern

class BeamShaper(PatternGenerator):
    def define_params(self):
        self.fetcher = util.CameraImageFetcher(self.puzzle)
        pzp.param.spinbox(self, "Period (px)", 29, 1, 1023)(None)
        pzp.param.checkbox(self, "Use binary", False)(None)
        pzp.param.text(self, "Function to display", "1")(None)
        pzp.param.checkbox(self, "Sum along axis 0", False)(None)
        pzp.param.text(self, "Save file name", "woo.csv")(None)
    
    def define_actions(self):
        @pzp.action.define(self, "Set background image")
        def set_background():
            self.fetcher.set_backbround()

        @pzp.action.define(self, "Generate pattern")
        def generate():
            self.check_slm_status()
            period = self["Period (px)"].value
            binary = self.puzzle["BinaryGrating"]
            blazed = self.puzzle["BlazedGrating"]
            use_binary = self["Use binary"].value
            if use_binary:
                binary[pat.BinaryGrating.PARAM_DUTY_CYCLE].set_value(0.5)
                binary[pat.BinaryGrating.PARAM_PERIOD].set_value(period)
                binary[pat.BinaryGrating.PARAM_PHASE].set_value(1023)
                binary.actions[pat.BinaryGrating.ACTION_SEND]()
            else:
                blazed["Period"].set_value(period)
                blazed["Max phase"].set_value(1023)
                blazed.actions[PatternGenerator.ACTION_SEND]()
            
            image = self.puzzle["SLM"]["image"].value.astype(float)
            envelope_x = np.arange(0, image.shape[0])
            envelope = np.array(eval(self["Function to display"].value, {"x": envelope_x, "np": np}))
            if np.any(envelope < 0) or np.max(envelope) > 1:
                raise ValueError("Returned function contains invalid value")
            
            if use_binary:
                phase_multiplier = np.arccos(np.sqrt(envelope)) / np.pi
            else:
                raise RuntimeError("Not yet supported")
            
            image = (phase_multiplier * image.T).T
            self.send_image_to_slm(image.astype(int))

            X = 60
            beam_distrib = np.exp(-0.5*((envelope_x - np.max(envelope_x)/2) / X)**2)
            phase_mask = np.exp(1j*2*np.pi*phase_multiplier)
            expected = np.fft.ifftshift(np.fft.ifft(beam_distrib * phase_mask))
            plt.scatter(np.arange(expected.size), np.abs(expected)**2)
            plt.title("Expected")
            plt.show()

        @pzp.action.define(self, "Analyze result")
        def analyze():
            camera_img = self.fetcher.get_processed_image()
            integrated = np.sum(camera_img, axis=0 if self["Sum along axis 0"].value else 1)
            np.savetxt(self["Save file name"].value, integrated)
            plt.scatter(np.arange(0, integrated.size), integrated)
            plt.xlabel("Position on camera (px)")
            plt.ylabel("Intensity")
            plt.show()