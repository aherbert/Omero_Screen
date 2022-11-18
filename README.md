# Program to analyse Omero high-throughput microscopy images

This is a beta version of the program; additional bug fixes and testing necessary
Requires a connection to an active Omero server

## Program lay out

The software logs onto the Omero server (default set to uni of Sussex server)
via a decorator (in /omero_screen/general_functions)
It collects metadata from an excel file (default sample file provided in /data)
and stored this information in the Experiment_DATA class. 


