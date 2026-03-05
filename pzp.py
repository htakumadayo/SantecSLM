import puzzlepiece as pzp
from puzzlepiece.extras import hardware_tools as pht
from pzp_hardware.generic.mixins import image_preview
import numpy as np
import SantecSLM.interface as itf
from qtpy.QtWidgets import QApplication, QFileDialog
import pandas as pd


""" Puzzlepiece pieces designed for Santec SLM(especially 200)"""

dummy_SLM_dimension = (1200, 1920)  # (height, width)


# Dict key of Puzzle.globals for SLM api object.
SANTEC_SLM_API = "SANTEC_SLM200_API"

class SLMPiece(image_preview.ImagePreview, pzp.Piece):
    live_toggle = False # include a toggle to enable live mode
    autolevel_toggle = False # include a toggle to enable autoleveling
    max_counts = 1023 # the maximum image brightness (will be white with autolevel off)
    use_numba = False # if you need increased perfo
    horizontal_layout = True

    # String constants to access piece parameters and actions
    PARAM_DISPLAY_NB = "SLM Display number"
    PARAM_CTRL_NB = "SLM number"
    PARAM_WAVELENGTH = "Wavelength (nm)"
    PARAM_PHASE = "Phase"   # Maybe put an explicit description for this param
    PARAM_SLM_DIMENSIONS = "SLM dimensions"
    PARAM_WL_SAVE = "Save permanently λ and phase on update"
    PARAM_IMAGE = "image"
    PARAM_CONNECTED = "connected"

    ACTION_UPDATE_WL = "Update target wavelength and phase"
    ACTION_CUSTOM_PAT = "Load custom pattern (csv)"

    def define_params(self):
        super().define_params()
        
        pzp.param.spinbox(self, SLMPiece.PARAM_DISPLAY_NB, 2, 1, 10, visible=False)(None)
        pzp.param.spinbox(self, SLMPiece.PARAM_CTRL_NB, 1, 1, 8, visible=False)(None)
        pzp.param.spinbox(self, SLMPiece.PARAM_WAVELENGTH, 635, 400, 1400, visible=False)(None)
        pzp.param.spinbox(self, SLMPiece.PARAM_PHASE, 200, 0, 999, visible=False)(None)
        pzp.param.checkbox(self, SLMPiece.PARAM_WL_SAVE, 1, visible=False)(None)

        pzp.param.array(self, SLMPiece.PARAM_SLM_DIMENSIONS, visible=False)(None)

        @pzp.param.connect(self)
        def connect():
            if self.puzzle.debug:
                self[SLMPiece.PARAM_SLM_DIMENSIONS].set_value(np.array(dummy_SLM_dimension))
                return 1
            
            # If not in debug mode
            slm: itf.SLM = self.puzzle.globals[SANTEC_SLM_API]
            display_nb = self[SLMPiece.PARAM_DISPLAY_NB].value
            slm_nb = self[SLMPiece.PARAM_CTRL_NB].value

            itf.check_error(slm.SLM_Disp_Open(display_nb))
            itf.check_error(slm.SLM_Ctrl_Open(slm_nb))

            rcode, width, height = slm.SLM_Disp_Info(display_nb)
            itf.check_error(rcode)

            rcode, wavelength, phase = slm.SLM_Ctrl_ReadWL(slm_nb)
            itf.check_error(rcode)

            self[SLMPiece.PARAM_SLM_DIMENSIONS].set_value(np.array((height, width)))
            self[SLMPiece.PARAM_WAVELENGTH].set_value(wavelength)
            self[SLMPiece.PARAM_PHASE].set_value(phase)
            return 1

        @pzp.param.disconnect(self)
        def disconnect():
            if not self.puzzle.debug:
                slm = self.puzzle.globals[SANTEC_SLM_API]
                itf.check_error(slm.SLM_Disp_Close(self[SLMPiece.PARAM_DISPLAY_NB].value))
                itf.check_error(slm.SLM_Ctrl_Close(self[SLMPiece.PARAM_CTRL_NB].value))
            return 0
        
        @pzp.param.array(self, self.PARAM_IMAGE)
        def image():
            return self[self.PARAM_IMAGE].value
        
        @image.set_setter(self)
        def image_setter(value):
            if self[self.PARAM_CONNECTED].value is not True:
                return None
            slm_dim = self[SLMPiece.PARAM_SLM_DIMENSIONS].value.astype(int)
            if slm_dim[0] != value.shape[0] or slm_dim[1] != value.shape[1]:
                raise ValueError(f"Pattern doesn't have dimension of SLM: SLM dim: {slm_dim}, pattern dim: {value.shape}")
            if np.min(value) < 0 or np.max(value) > 1023:
                raise ValueError(f"Pattern data contains invalid value. (Must be between 0 and 1023)")
            if not self.puzzle.debug:
                rcode = self.puzzle.globals[SANTEC_SLM_API].SLM_Disp_Data(
                    self[SLMPiece.PARAM_DISPLAY_NB].value,
                    value.shape[1], value.shape[0],
                    itf.FLAGS_COLOR_GRAY, value)
                itf.check_error(rcode)
            return value
    
    def define_actions(self):
        @pzp.action.define(self, SLMPiece.ACTION_UPDATE_WL, visible=False)
        def updateslm():
            if self.puzzle.debug:
                print("Update")
                return
            if self[self.PARAM_CONNECTED].value is not True:
                return
            wl = self[SLMPiece.PARAM_WAVELENGTH].value
            phase = self[SLMPiece.PARAM_PHASE].value
            slm_nb = self[SLMPiece.PARAM_CTRL_NB].value
            slm = self.puzzle.globals[SANTEC_SLM_API]
            itf.check_error(slm.SLM_Ctrl_WriteWL(slm_nb, wl, phase), "Write Wavelength:")
            
            if self[SLMPiece.PARAM_WL_SAVE].value:
                itf.check_error(slm.SLM_Ctrl_WriteAW(slm_nb), "Save Wavelength:")
        
        @pzp.action.define(self, SLMPiece.ACTION_CUSTOM_PAT, visible=True)
        def custom_pattern():
            if not self.puzzle.debug and not self[self.PARAM_CONNECTED].value:
                raise RuntimeError("SLM is not connected.")
            # Ensure a QApplication exists (create one only if needed)
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            path, _ = QFileDialog.getOpenFileName(
                None,
                "Select csv pattern",
                "",
                "CSV (*.csv);;All files (*)"
            )
            if not path:
                return
            data = pd.read_csv(path, sep=None, engine="python").to_numpy()
            self[self.PARAM_IMAGE].set_value(data)

        pzp.action.settings(self)

    def setup(self):
        if not self.puzzle.globals.require(SANTEC_SLM_API):
            # If there are multiple dlls, the user should specify only the directory and not all the dlls
            dll_path = "dll/x64/SLMFunc.dll"  # default
            dll_path = pht.config("santec_api_dll_path", 
                                  default=dll_path, 
                                  description="Santec SLM api dll", 
                                  validator=pht.validator_path_exists)
            self.puzzle.globals[SANTEC_SLM_API] = itf.SLM(dll_path)

if __name__ == "__main__":
    pass

