from omero_screen.loops import plate_loop
from omero_screen import Defaults
from omero_screen.general_functions import omero_connect


@omero_connect
def main(plate_id, options=None, conn=None):
    if options:
        Defaults.update(options)
    plate_loop(plate_id, conn)


if __name__ == "__main__":
    main(1430)
