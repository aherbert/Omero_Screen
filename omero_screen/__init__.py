# omero_screen/__init__.py

__version__ = "0.1.1"


import pathlib
import json
import logging


def setup_logging():
    print("Setting up logging")
    # Set a less verbose level for the root logger
    logging.basicConfig(level=logging.WARNING)

    # Create and configure your application's main logger
    app_logger_name = "omero-screen"
    app_logger = logging.getLogger(app_logger_name)
    app_logger.setLevel(logging.DEBUG)  # Set to DEBUG or any other level

    # Prevent propagation to the root logger
    app_logger.propagate = False

    # Create a console handler for the logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # Ensure it captures all levels processed by the logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    ch.setFormatter(formatter)
    app_logger.addHandler(ch)


# Ensure this is called when your package is imported
setup_logging()
logger = logging.getLogger("omero-screen")

# Derive the absolute path to the config.json file
current_dir = pathlib.Path(__file__).parent
config_path = current_dir / "../data/secrets/config.json"
try:
    with open(config_path) as file:
        server_data = True
        data = json.load(file)
        username = data["username"]
        password = data["password"]
        server = data["server"]
        project = data["project"]
        logger.info(f"Successfully loaded server data for {server}")
except IOError:
    server_data = False
    username = "invalid"
    password = "invalid"
    server = "invalid"
    project = 5313


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
        "RPE-1_WT_CycE": "RPE-1_Tub_Hoechst",
        "RPE-1_P53KO_CycE": "RPE-1_Tub_Hoechst",
        "HELA": "HeLa_Tub_Hoechst",
        "U2OS": "U2OS_Tub_Hoechst",
        "MM231": "RPE-1_Tub_Hoechst",
        "HCC1143": "RPE-1_Tub_Hoechst",
        "MM231_SCR": "MM231_Tub_Hoechst",
        "MM231_GWL": "MM231_Tub_Hoechst",
        "SCR_MM231": "MM231_Tub_Hoechst",
        "GWL_MM231": "MM231_Tub_Hoechst",
        "RPE1WT": "RPE-1_Tub_Hoechst",
        "RPE1P53KO": "RPE-1_Tub_Hoechst",
        "RPE1wt_PALB": "only_PALB",
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
    "SERVER_DATA": server_data,
    "USERNAME": username,
    "PASSWORD": password,
    "SERVER": server,
    "PROJECT_ID": project,
    "GPU": None,
    "MAGNIFICATION": "10x",
    "INFERENCE_MODEL": None,
    "INFERENCE_GALLERY_WIDTH": 0,
    "INFERENCE_BATCH_SIZE": 16,
}
