
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from omero_screen import Defaults  # noqa: E402
from omero_screen.loops import plate_loop  # noqa: E402
from omero_screen.general_functions import omero_connect  # noqa: E402
import argparse



@omero_connect
def main(plate_id, inference_model, options=None, conn=None):
    if options:
        Defaults.update(options)

    plate_loop(plate_id, conn, inference_model)


if __name__ == "__main__":
    Defaults['server'] = "../data/secrets/config_test.json"
    
    main(351)
