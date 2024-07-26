


import omero
from omero.gateway import BlitzGateway, _ImageWrapper, _DatasetWrapper
from ezomero import get_image
import numpy as np

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
    """
    Add a map annotation to an OMERO object.
    :param omero_object: OMERO object to annotate
    :param key_value: List of key-value pairs
    :param conn: OMERO connection
    """
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
    img, array = get_image(conn, image.getId())
    array_squeezed = np.squeeze(array, axis=0)
    return np.max(array_squeezed, axis=0)


def image_generator(image_array):
    for c in range(image_array.shape[-1]):
        yield image_array[..., c]


def load_mip(conn: BlitzGateway, image: _ImageWrapper, dataset_id: int) -> None:
    """
    Load the maximum intensity projection of a z-stack image to OMERO.
    :param conn: OMERO connection
    :param image: _ImageWrapper object
    :param dataset: _DatasetWrapper object
    :return: None
    """
    dataset = conn.getObject("Dataset", dataset_id)
    mip = process_mip(conn, image)
    channel_num = mip.shape[-1]
    mip_name = f"mip_{image.getId()}"
    img_gen = image_generator(mip)
    new_image = conn.createImageFromNumpySeq(
        img_gen, mip_name, 1, channel_num, 1, dataset=dataset
    )
    add_map_annotation(image, [("name", f"mip_{new_image.getId()}")], conn=conn)
    return int(new_image.getId())


def parse_mip(image_id, dataset_id, conn):
    image = conn.getObject("Image", image_id)
    if not (mip_id := check_map_annotation(image)):
        return load_mip(conn, image, dataset_id)
    return int(mip_id)



if __name__ == "__main__":
    mip = parse_mip()
    print(mip)
    
    