import omero.gateway

from omero_screen.metadata import MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.omero_functions import delete_map_annotations
from tests.conftest import attach_excel_file, delete_excel_attachments, delete_object

plate_id = 2


def test_flatfield_corr_create(omero_conn):
    """
    Tests the flatfield_corr function.
    """
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    attach_excel_file(plate_obj, "./data/metadata.xlsx", omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    assert list(flatfield_dict.keys()) == [
        "DAPI",
        "Tub",
        "p21",
        "EdU",
    ], "Flatefield dictionary does not contain the correct keys"
    assert flatfield_dict["DAPI"].shape == (
        1080,
        1080,
    ), "Flatefield dictionary does not contain the correct keys"
    delete_excel_attachments(plate_id, omero_conn)
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)


def test_example_image_attach(omero_conn):
    datasets = omero_conn.getObjects("Dataset", attributes={"name": str(plate_id)})
    dataset = list(datasets)[0]
    annotations = list(dataset.listAnnotations())
    pdf_annotations = [
        ann
        for ann in annotations
        if isinstance(ann, omero.gateway.FileAnnotationWrapper)
        and ann.getFile().getName().endswith(".pdf")
    ]
    pdf_list = [annotation.getFile().getName() for annotation in pdf_annotations]
    assert len(pdf_annotations) == 4, "Example Images were not attached to the data set"
    assert (
        "DAPI_flatfield_check.pdf" in pdf_list
    ), "Example Images are not named correctly"


def test_flatfield_corr_load(omero_conn):
    """
    Tests the flatfield_corr function.
    """
    # load metadata excel file
    plate_obj = omero_conn.getObject("Plate", plate_id)
    attach_excel_file(plate_obj, "./data/metadata.xlsx", omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    assert list(flatfield_dict.keys()) == [
        "DAPI",
        "Tub",
        "p21",
        "EdU",
    ], "Flatefield dictionary does not contain the correct keys"
    assert flatfield_dict["DAPI"].shape == (
        1080,
        1080,
    ), "Flatefield dictionary does not contain the correct keys"
    delete_excel_attachments(plate_id, omero_conn)
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)
    delete_object(omero_conn, "Dataset", project_setup.dataset_id)
    delete_object(omero_conn, "Project", project_setup.project_id)
