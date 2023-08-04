from omero_screen.database_links import MetaData, ProjectSetup
from omero_screen.flatfield_corr import flatfieldcorr
from omero_screen.image_analysis_nucleus import NucImage, NucImageProperties
from omero_screen.omero_functions import delete_map_annotations
from tests.conftest import attach_excel_file, delete_excel_attachments, delete_object
plate_id= 2
def test_image_data(omero_conn):
    plate_obj = omero_conn.getObject("Plate", plate_id)
    well = omero_conn.getObject("Well", 49)
    omero_image = well.getImage(0)
    attach_excel_file(plate_obj, './data/metadata.xlsx', omero_conn)
    metadata = MetaData(omero_conn, plate_id=plate_id)
    project_setup = ProjectSetup(plate_id, omero_conn)
    flatfield_dict = flatfieldcorr(metadata, project_setup, omero_conn)
    image = NucImage(omero_conn, well, omero_image, metadata, project_setup, flatfield_dict)
    image_data = NucImageProperties(well, image, metadata)
    image_df = image_data.image_df
    assert round(image_df.loc[0,'intensity_max_DAPI_nucleus'].mean(), 3) == 16563.216, "Nuclear Segmentation is incorrect"
    assert len(image_df) == 190, "The length of the image df is incorrect"
    assert round(image_data.image_df['intensity_mean_EdU_nucleus'].mean(), 3) == 3619.601, "The mean of the EdU intensity is incorrect"
    assert round(image_data.quality_df['intensity_median'].mean(),3) == 1841.78, "The median of the EdU intensity in the quality df is incorrect"
    delete_excel_attachments(plate_id, omero_conn)
    delete_map_annotations(plate_obj, omero_conn)
    for well in plate_obj.listChildren():
        delete_map_annotations(well, omero_conn)
    delete_object(omero_conn, "Dataset", project_setup.dataset_id)
    delete_object(omero_conn, "Project", project_setup.project_id)



