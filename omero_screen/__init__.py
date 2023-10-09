# omero_screen/__init__.py

__version__ = "0.1.1"


import pathlib

Defaults = {
    "DEFAULT_DEST_DIR": str(
        pathlib.Path.home() / "Desktop"
    ),  # Decides where the final data folder will be made
    "DEFAULT_SUMMARY_FILE": "screen-dir",  # Record the result directory for the screen
    "FLATFIELD_TEMPLATES": "flatfield_correction_images",
    "DATA": "single_cell_data",
    "QUALITY_CONTROL": "quality_control",
    "IMGS_CORR": "images_corrected",
    "TEMP_WELL_DATA": "temp_well_data",
    "PLOT_FIGURES": "figures",
    "DATA_CELLCYCLE_SUMMARY": "cellcycle_summary",
    "MODEL_DICT": {
        "nuclei": "Nuclei_Hoechst",
        "RPE-1": "RPE-1_Tub_Hoechst",
        "RPE-1_WT": "RPE-1_Tub_Hoechst",
        "RPE-1_P53KO": "RPE-1_Tub_Hoechst",
        "HELA": "HeLa_Tub_Hoechst",
        "U2OS": "U2OS_Tub_Hoechst",
        "MM231": "RPE-1_Tub_Hoechst",
        "MM231_SCR": "MM231_Tub_Hoechst",
        "MM231_GWL": "MM231_Tub_Hoechst",
        "SCR_MM231": "MM231_Tub_Hoechst",
        "GWL_MM231": "MM231_Tub_Hoechst",
    },
    "NS": "openmicroscopy.org/omero/client/mapAnnotation",
    "FEATURELIST": [
        "label",
        "area",
        "intensity_max",
        "intensity_min",
        "intensity_mean",
        "centroid",
    ],
    "DEBUG": False,
    "PROJECT_ID": 5313,
    "GPU": None,
}
