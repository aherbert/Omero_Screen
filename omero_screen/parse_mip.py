

import logging
import omero
from omero.gateway import BlitzGateway, _ImageWrapper
from ezomero import get_image
import numpy as np
from omero_screen.omero_functions import remove_map_annotations

logger = logging.getLogger("omero-screen")
def check_map_annotation(omero_object):
    """
    Check if a map annotation with a specific key exists.
    :param omero_object: OMERO object to check
    :param key: Key to check
    :param conn: OMERO connection
    :return: True if the key exists, False otherwise
    """
    if map_anns := omero_object.listAnnotations(
        ns=omero.constants.metadata.NSCLIENTMAPANNOTATION
    ):
        for ann in map_anns:
            ann_values = dict(ann.getValue())
            for item in ann_values.values():
                print(item)
                if "mip" in item:
                    return item.split("_")[-1]
    return None



def add_map_annotation(omero_object, key_value, conn=None):
    # sourcery skip: use-named-expression
    """
    Add a map annotation to an OMERO object.
    :param omero_object: OMERO object to annotate
    :param key_value: List of key-value pairs
    :param conn: OMERO connection
    """
    
    map_anns = list(
        omero_object.listAnnotations(ns=omero.constants.metadata.NSCLIENTMAPANNOTATION)
    )
    if map_anns:  # If there are existing map annotations
        for ann in map_anns:
            ann_values = dict(ann.getValue())
            if any("mip" in str(value) for value in ann_values.values()):
                conn.deleteObject(ann._obj)  # Delete the annotation
                
    map_ann = omero.gateway.MapAnnotationWrapper(conn) 
    map_ann.setNs(omero.constants.metadata.NSCLIENTMAPANNOTATION)
    map_ann.setValue(key_value)
    map_ann.save()
    omero_object.linkAnnotation(map_ann)


def process_mip(conn: BlitzGateway, image: _ImageWrapper) -> np.ndarray:
    """
    Generate maximum intensity projection of a z-stack image.
    The get_image function returns an array of the shape (t, z, x, y, c).
    This function only takes arrays with a single time dimension, which gets
    eliminated by the squeeze function.
    :param conn: OMERO connection
    :param image: _ImageWrapper object
    :return: numpy array of maximum intensity projection (x, y, c)
    """
    _, array = get_image(conn, image.getId())
    max_array = np.max(array, axis=1)
    #shape will be t, x, y, c and needs to expand to t, 1, x, y, c
    return np.expand_dims(max_array, axis=1)


def image_generator(image_array):
    flattened_array = np.squeeze(image_array)
    print(f"flattened_array shape is {flattened_array.shape}")
    for c in range(flattened_array.shape[-1]):
        yield flattened_array[..., c]


def load_mip(conn: BlitzGateway, image: _ImageWrapper, dataset_id: int) -> None:
    """
    Load the maximum intensity projection of a z-stack image to OMERO.
    :param conn: OMERO connection
    :param image: _ImageWrapper object
    :param dataset: _DatasetWrapper object
    :return: None
    """
    dataset = conn.getObject("Dataset", dataset_id)
    mip_array = process_mip(conn, image)
    print(f"mip_array shape is {mip_array.shape}")
    channel_num = mip_array.shape[-1]
    mip_name = f"mip_{image.getId()}"
    img_gen = image_generator(mip_array)
    new_image = conn.createImageFromNumpySeq(
        img_gen, mip_name, 1, channel_num, 1, dataset=dataset
    )
    add_map_annotation(image, [("max_int_projection", f"mip_{new_image.getId()}")], conn=conn)
    return mip_array


def parse_mip(image_id, dataset_id, conn):
    image = conn.getObject("Image", image_id)

    mip_id = check_map_annotation(image)
    if not mip_id:
        return load_mip(conn, image, dataset_id)
    _, mip_array = get_image(conn, int(mip_id))
    if isinstance(mip_array, np.ndarray):
        return mip_array
    # remove map annotation if the mip is missing
    # map_anns = [
    #     ann
    #     for ann in image.listAnnotations()
    #     if isinstance(ann, omero.gateway.MapAnnotationWrapper)
    # ]
    # for ann in map_anns:
    #     ann_values = dict(ann.getValue())
    #     for item in ann_values.values():
    #             if "mip" in item:
    #                 print(f"Removing annotation {item}")
    #                 conn.deleteObjects(ann._obj)

    # image.save()

    logger.warning("The image is linked to MISSING MIP")
    return load_mip(conn, image, dataset_id)
        


if __name__ == "__main__":
    conn = BlitzGateway("root", "omero", host="localhost")
    conn.connect()
    mip = parse_mip(8151, 452, conn)
    print(mip.mean())
    conn.close()
    
    