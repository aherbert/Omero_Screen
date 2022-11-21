# Program to analyse Omero high-throughput microscopy images

This is a beta version of the program; additional bug fixes and testing necessary
Requires a connection to an active Omero server



## Contributors
Robert Zach, Haoran Yue, Alex Herbert and Helfrid Hochegger

## Purpose of the program

Analysing hightroughput imaging data using Omero Screen data base.

Images are first pre-processed using flatfield correction, then segmenthed using optimised cellpose models.
Single cell features (default,: area, min max intensities) for each channel are extracted and stored in a final data frame.

The input metadata are provided via an Excel file (see data/sample_metadata.xlsx)


### TODO Generate module to make Figures for cell cycle data
### TODO Add omero screen metadata storage functionality
### TODO use classifier for mitotic interphase separation and free up H3 channel

