# conftest.py
import pytest
import omero
from omero.gateway import BlitzGateway

@pytest.fixture(scope='session')
def omero_conn():
    # Create a connection to the omero server
    conn = BlitzGateway(
        'root',
        'omero',
        host='localhost',
        port=4064
    )

    conn.connect()
    print("Connected to OMERO.server")

    yield conn

    # After the tests are done, disconnect
    conn.close()
    print("Closed connection to OMERO.server")


def delete_object(conn, object_type, object_id):
    """
    Delete an object of the given type and ID.
    """
    delete = omero.cmd.Delete2(targetObjects={object_type: [object_id]})
    cb = conn.c.sf.submit(delete)
    cb = omero.callbacks.CmdCallbackI(conn.c, cb)

    # Wait for the delete to finish
    while cb.block(500) is False:
        pass

    rsp = cb.getResponse()
    if isinstance(rsp, omero.cmd.OK):
        print(f"Deleted {object_type} {object_id}")
    else:
        print(f"Failed to delete {object_type} {object_id}")
