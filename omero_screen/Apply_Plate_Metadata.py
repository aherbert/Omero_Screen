#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------
"""
Applies metadata from a file to HCS plate data.
"""

import re
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_GB')
except:
    pass
import pandas as pd
import numbers
from collections import OrderedDict

import omero.scripts as scripts
from omero.gateway import BlitzGateway
from omero.rtypes import *  # noqa

PARAM_DATATYPE = "Data_Type"
PARAM_IDS = "IDs"
PARAM_DELETE_FILE = "Delete_File"

# Use the OME namespace so the tag can be edited in Insight
NS = 'openmicroscopy.org/omero/client/mapAnnotation'

# Regular expression for files supported by pandas.read_excel
# This is for Pandas 1.1.5 that is for Python 3.6.
# Later versions require Python 3.8 which is not used by the OMERO 5.6.3 server.
# Note that newer Excel formats (.xlsx) have security vunerabilities that
# require an upgrade to Python 3.8+ with Pandas using openpyxl.
SHEET_PATTERN = re.compile('\.(xls|ods)$', re.IGNORECASE)

###############################################################################

def split_well_label(label):
    """
    Splits the well label into (row, column).
    Assumes all leading letters are the row and the column is only digits
    @param label: Well label
    @return (row, column)
    """
    for i, c in enumerate(label):
        if c.isdigit() and label[i:].isdigit():
            return (label[:i], int(label[i:]))
    # Unknown format
    return (label,'')


def error(msg):
    """
    Print an error message
    @param msg: The message
    """
    print("ERROR:", msg)


def process_plate(conn, plate_id, meta):
    """
    Process the metadata for the plate.

    @param conn:     The BlitzGateway connection
    @param plate_id: The plate Id
    @param meta:     The metadata
    """
    # Allow all groups
    conn.SERVICE_OPTS.setOmeroGroup(-1)
    plate = conn.getObject('Plate', plate_id)
    if not plate:
        error('Missing plate %s' % plate_id)
        return 0

    links = []

    # Create annotations in the correct group.
    # This currently does not check if a user has permissions to annotate
    # the objects. Incorrect permissions will raise an error during save.
    gid = plate.getDetails().getGroup().getId()
    ctx = conn.SERVICE_OPTS
    ctx.setOmeroGroup(gid)

    # Plate metadata
    kv = meta.get('Plate')
    if kv:
        ann = plate.getAnnotation(NS)
        if ann:
            # Update existing
            d = OrderedDict(ann.getValue())
            for (k, v) in kv:
                d[k] = v
            value = list(d.items())
            if value != ann.getValue():
                ann.setValue(value)
                ann.save()
        else:
            # New
            tag = omero.gateway.MapAnnotationWrapper(conn)
            tag.setValue(kv)
            tag.setNs(NS)
            tag.save()
            # Create a link from the plate to the annotation
            link = omero.model.PlateAnnotationLinkI()
            link.parent = omero.model.PlateI(plate.getId(), False)
            link.child = omero.model.MapAnnotationI(tag.getId(), False)
            links.append(link)

    # Well metadata
    wd = meta.get('Well')
    if wd:
        rl = plate.getRowLabels()
        cl = plate.getColumnLabels()
        wg = plate.getWellGrid()

        for (w, kv) in wd.items():
            (row, col) = split_well_label(w)
            r = rl.index(row) if row in rl else None
            c = cl.index(col) if col in cl else None
            if r is not None and c is not None:
                well = wg[r][c]
                if well is None:
                    error("Missing well " + w)
                    continue

                ann = well.getAnnotation(NS)
                if ann:
                    # Update existing
                    d = OrderedDict(ann.getValue())
                    for (k, v) in kv:
                        d[k] = v
                    value = list(d.items())
                    if value != ann.getValue():
                        ann.setValue(value)
                        ann.save()
                else:
                    # New
                    tag = omero.gateway.MapAnnotationWrapper(conn)
                    tag.setValue(kv)
                    tag.setNs(NS)
                    tag.save()
                    # Create a link from the well to the annotation
                    link = omero.model.WellAnnotationLinkI()
                    link.parent = omero.model.WellI(well.getId(), False)
                    link.child = omero.model.MapAnnotationI(tag.getId(), False)
                    links.append(link)
            else:
                error("Unknown well " + w)

    # Save all links together
    if links:
        try:
            savedLinks = conn.getUpdateService().saveAndReturnArray(links, ctx)
        except omero.ValidationException as x:
            error("Failed to create links: %s" % x)
            return 0

    return 1


def run(conn, params):
    """
    Process all attached files for objects defined in the script parameters
    into plate metadata.

    @param conn:   The BlitzGateway connection
    @param params: The script parameters
    """

    seen = set()
    count = 0
    ok = 0
    for i in params.get(PARAM_IDS, []):
        if i in seen:
            continue
        seen.add(i)
        obj = conn.getObject(params[PARAM_DATATYPE], i)
        if not obj:
            error("Unknown %s: %s" % (params[PARAM_DATATYPE], i))
            continue

        # Find all metadata files
        for ann in obj.listAnnotations():
            if isinstance(ann, omero.gateway.FileAnnotationWrapper):
                if SHEET_PATTERN.search(ann.getFileName()):
                    print("Processing %s %s file %s" %
                          (params[PARAM_DATATYPE], i, ann.getFileName()))
                    count = count + 1
                    f = ann.getFile()
                    with f.asFileObj() as fo:
                        (plate_id, meta) = read_spreadsheet(fo)

                    if meta is None:
                        error("Failed to read %s %s file %s" %
                              (params[PARAM_DATATYPE], i, ann.getFileName()))
                        continue

                    if params[PARAM_DATATYPE] == 'Screen':
                        if not plate_id:
                            error("Unknown Plate for Screen %s file %s" %
                                  (i, ann.getFileName()))
                            continue
                    else:
                        # If it was read, check the plate ID matches
                        if plate_id and plate_id != i:
                            error("Plate ID mismatch for Plate %s file %s" %
                                  (i, ann.getFileName()))
                            continue
                        plate_id = i

                    if process_plate(conn, plate_id, meta):
                        # Optionally delete file
                        if params.get(PARAM_DELETE_FILE) and f.canDelete():
                            try:
                                conn.deleteObject(
                                    omero.model.OriginalFileI(f.getId()))
                            except Exception as e:
                                error("Failed to remove %s %s file %s : %s" %
                                      (params[PARAM_DATATYPE], i,
                                       ann.getFileName(), e))
                                continue
                        ok = ok + 1

    return (count, ok)


def summary(count, ok):
    """
    Produce a summary message of the plate sets and error count

    @param count:  Total count of processed plates
    @param ok:     Count of plates processed without error
    """
    error_count = count - ok
    msg = "%d plate%s : %s error%s" % (
        ok, ok != 1 and 's' or '',
        error_count, error_count != 1 and 's' or '')
    return msg


def read_spreadsheet(io):
    """
    Read metadata from the spreadsheet.

    The plate ID is zero if it cannot be read from the 'Plate_ID' sheet,
    or an exception occurred.

    The metadata is returned using the keys ['Plate', 'Well'].

    The 'Plate' key has a list of (key,value) tuples.

    The 'Well' key has a dictionary keyed by well ID (e.g. 'A1', 'A2').
    Each well entry is a list of (key,value) tuples.

    @param io:  Spreadsheet input
    @return (plate_id, metadata)
    """
    plate_id = 0
    meta = dict()
    try:
        # Read all worksheets
        data = pd.read_excel(io, None)

        # Plate ID
        # Get Plate ID sheet
        if 'Plate_ID' in data:
            df = data.get('Plate_ID')
            # Get Plate_ID column
            if 'Plate_ID' in df:
                s = df.get('Plate_ID')
                # Get first entry
                if isinstance(s.get(0), numbers.Integral):
                    plate_id = long(s.get(0)) or 0

        # Plate data
        if 'Channels' in data:
            df = data.get('Channels')
            # Channels, Index
            k = df.keys()
            cols = ['Channels', 'Index']
            if all(elem in df.keys() for elem in cols):
                df = df[cols]
                pl = meta.setdefault('Plate', [])
                for i, row in df.iterrows():
                    pl.append((str(row[0]), str(row[1])))

        # Well data
        if 'Plate_Layout' in data:
            df = data.get('Plate_Layout').dropna()
            # Well, col1, col2, ...
            # Remove redundant Well_ID column
            k = df.keys().drop('Well_ID', errors='ignore')
            if 'Well' in k:
                k = k.drop('Well')
                wd = meta.setdefault('Well', dict())
                for i, row in df.iterrows():
                    w = row['Well']
                    d = wd.setdefault(w, [])
                    for x in k:
                        d.append((str(x), str(row[x])))

        # Field data ...

    except Exception as e:
        print("An error occurred: %s" % e)
        return (0, None)

    return (plate_id, meta)


def run_with_files():
    """
    Function to allow the script to be called outside of the OMERO
    scripting environment as a program to process input metadata files.
    """
    from optparse import OptionParser, OptionGroup
    import getpass
    import json

    parser = OptionParser(usage='usage: %prog [options] file.xls [file2.xls ...]',
                          description='Program to apply metadata to plates',
                          add_help_option=True, version='%prog 1.0')

    parser.add_option('--dry-run', dest='dry_run',
                     action='store_true', default=False,
                     help='Process the metadata but do not apply to plates')
    parser.add_option('-v', '--verbosity', dest='verbosity',
                     action='count', default=0,
                     help='Verbosity')

    group = OptionGroup(parser, 'OMERO')
    group.add_option('-u', '--username', dest='username',
                     help='OMERO username (will prompt if missing)')
    group.add_option('-p', '--password', dest='password',
                     help='OMERO password (will prompt if missing)')
    group.add_option('--host', dest='host', default='localhost',
                     help='OMERO host [%default]')
    group.add_option('--port', dest='port', default=4064,
                     help='OMERO port [%default]')

    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    # Process each metadata file.
    data = []

    for arg in args:
        if SHEET_PATTERN.search(arg):
            if options.verbosity:
                print('Reading ' + arg);
            (plate_id, meta) = read_spreadsheet(arg)
            if plate_id:
                data.append([plate_id, meta])
            else :
                print('Missing plate ID: ' + arg)
        else:
            print('Unknown input metadata format: ' + arg)

    if options.verbosity > 1:
        print(json.dumps(data, indent=2))

    if options.dry_run:
        return

    # Connect to OMERO
    if not options.username:
        options.username = input("OMERO username: ")
    if not options.password:
        options.password = getpass.getpass("OMERO password: ")
    conn = BlitzGateway(options.username, options.password,
                        host=options.host, port=options.port)
    try:
        if not conn.connect():
            raise Exception("Failed to connect to OMERO: %s" %
                            conn.getLastError())

        # Pass each input metadata to a function to process the plate.
        ok = 0
        for (plate_id, meta) in data:
            if options.verbosity:
                print('Processing plate %d' % plate_id);
            if process_plate(conn, plate_id, meta):
                ok = ok + 1

        print(summary(len(data), ok))
    except Exception as e:
        print("An error occurred: %s" % e)
    finally:
        conn.close()


def run_as_program():
    """
    Testing function to allow the script to be called outside of the OMERO
    scripting environment. The connection details and image ID must be valid.
    """

    import getpass
    username = input("OMERO username: ")
    password = getpass.getpass("OMERO password: ")
    host = 'localhost'
    port = 4064
    h = input("OMERO host (%s): " % host)
    if h:
        host = h
    p = input("OMERO port (%d): " % port)
    if p:
        port = p

    conn = BlitzGateway(username, password, host=host, port=port)
    try:
        if not conn.connect():
            raise Exception("Failed to connect to OMERO: %s" %
                            conn.getLastError())

        params = {}
        #params[PARAM_DATATYPE] = "Screen"
        #params[PARAM_IDS] = [101]
        params[PARAM_DATATYPE] = "Plate"
        params[PARAM_IDS] = [51]
        params[PARAM_DELETE_FILE] = True

        (count, ok) = run(conn, params)

        print(summary(count, ok))
    except Exception as e:
        print("An error occurred: %s" % e)
    finally:
        conn.close()


def run_as_script():
    """
    The main entry point of the script, as called by the client via the
    scripting service, passing the required parameters.
    """
    dataTypes = [wrap('Screen'), wrap('Plate')]

    client = scripts.client('Apply_Plate_Metadata.py', """\
Applies metadata from attached file(s) to HCS plate data.

Metadata should be in a Excel spreadsheet format (.xls, .ods).
Each file for the selected IDs is processed. Metadata has the following format:

Sheet: Column1, Column2, ...

Plate_ID: Plate_ID
Plate_Layout: Well, Column1, Column2, ...
Channels: Channels, Index

The Plate_ID data is used to identify the plate if the metadata file is
attached to a Screen. If the metadata file is attached to a Plate then
this sheet is not required.

The Plate_Layout data is applied to each well.
The Channels data is applied to each plate.

Warning: Key-Value metadata is consolidated using unique keys. This will
overwrite existing values for the same key and consolidate duplicate
existing keys into a single entry.
""",  # noqa

    scripts.String(PARAM_DATATYPE, optional=False, grouping="1.1",
        description="The data you want to work with.", values=dataTypes,
        default="Plate"),

    scripts.List(PARAM_IDS, optional=True, grouping="1.2",
        description="List of Screen IDs or Plate IDs").ofType(rlong(0)),

    scripts.Bool(PARAM_DELETE_FILE, grouping="1.3",
        description="Delete processed metadata files",
        default=True),

    version="1.0",
    authors=["Alex Herbert", "GDSC"],
    institutions=["University of Sussex"],
    contact="a.herbert@sussex.ac.uk",
    )  # noqa

    try:
        conn = BlitzGateway(client_obj=client)

        # Process the list of args above.
        params = {}
        for key in client.getInputKeys():
            if client.getInput(key):
                params[key] = unwrap(client.getInput(key))
                
        # Call the main script - returns the screen ID
        (count, ok) = run(conn, params)

        msg = summary(count, ok)
    except Exception as e:
        msg = "An error occurred: %s" % e
    finally:
        print(msg)
        client.setOutput("Message", rstring(msg))
        client.closeSession()

if __name__ == "__main__":
    """
    Python entry point
    """
    function_to_run = run_as_script

    # Allow the script to be run on the command-line by passing the param 'run'
    # or by passing metadata files to process
    p2 = re.compile('^-h$|-help$')
    for arg in sys.argv:
        if arg == 'run':
            function_to_run = run_as_program
            break
        if SHEET_PATTERN.search(arg) or p2.search(arg):
            function_to_run = run_with_files
            break

    function_to_run()
