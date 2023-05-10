# omero_screen/__init__.py

__version__ = '0.1.1'


import pathlib
Defaults = {
'DEFAULT_DEST_DIR': pathlib.Path.home() / "Desktop",  # Decides where the final data folder will be made
'FLATFIELD_TEMPLATES': "flatfield_correction_images",
'DATA': "single_cell_data",
'QUALITY_CONTROL': "quality_control",
'IMGS_CORR': "images_corrected",
'TEMP_WELL_DATA': "temp_well_data",
'PLOT_FIGURES': "figures",
'DATA_CELLCYCLE_SUMMARY': "cellcycle_summary",
'PATH': pathlib.Path.cwd().parent,
'MODEL_DICT': {
    'nuclei': 'Nuclei_Hoechst',
    'RPE-1': 'RPE-1_Tub_Hoechst',
    'HELA': 'HeLa_Tub_Hoechst',
    'U2OS': 'U2OS_Tub_Hoechst',
    'MM231': 'MM231_Tub_Hoechst',
    'MM231_SCR': 'MM231_Tub_Hoechst',
    'MM231_GWL': 'MM231_Tub_Hoechst',
    'SCR_MM231': 'MM231_Tub_Hoechst',
    'GWL_MM231': 'MM231_Tub_Hoechst',
},
'NS': 'openmicroscopy.org/omero/client/mapAnnotation',
'FEATURELIST': ['label', 'area', 'intensity_max', 'intensity_mean'],
'DEBUG': False
}


SEPARATOR = "==========================================================================================\n"
