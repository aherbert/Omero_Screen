# Correct usage in another module
import omero_screen # This import should run setup_logging()
import logging

logger = logging.getLogger("omero-screen")
logger.info("This is a test log message.")