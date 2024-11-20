
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
from omero_screen import Defaults  # noqa: E402
from omero_screen.loops import plate_loop  # noqa: E402
from omero_screen.general_functions import omero_connect  # noqa: E402
import argparse



@omero_connect
def main(plate_id, options=None, conn=None):
    if options:
        Defaults.update(options)

    parser = argparse.ArgumentParser(description="Image analysis with optional inference.")
    parser.add_argument("--inference", type=str, metavar="MODEL", 
                        help="Run inference on the dataset using a pre-trained model. Specify the model filename.")
    args = parser.parse_args()

    plate_loop(plate_id, conn, args)


if __name__ == "__main__":
    Defaults['server'] = "../data/secrets/config_test.json"
    
    main(351)
