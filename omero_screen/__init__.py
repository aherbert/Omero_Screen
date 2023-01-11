# omero_screen/__init__.py

__version__ = '0.1.1'


import pathlib
class Defaults:
    """Store the default variables to read the Excel input file"""
    DEFAULT_DEST_DIR = "Desktop"  # Decides where the final data folder will be made
    FLATFIELD_TEMPLATES = "flatfield_correction_images"
    DATA = "single_cell_data"
    QUALITY_CONTROL = "quality_control"
    IMGS_CORR = "images_corrected"
    TEMP_WELL_DATA = "temp_well_data"
    PATH = pathlib.Path.cwd().parent
    MODEL_DICT = {
        'nuclei': 'Nuclei_Hoechst',
        'RPE-1': 'RPE-1_Tub_Hoechst',
        'HELA': 'HeLa_Tub_Hoechst',
        'U2OS': 'U2OS_Tub_Hoechst',
        'MM231': 'MM231_Tub_Hoechst',
        'MM231_SCR': 'MM231_Tub_Hoechst',
        'MM231_GWL': 'MM231_Tub_Hoechst',
    }
    NS = 'openmicroscopy.org/omero/client/mapAnnotation'
    FEATURELIST = ['label', 'area', 'intensity_max', 'intensity_mean']

SEPARATOR = "==========================================================================================\n"
