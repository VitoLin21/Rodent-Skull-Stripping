from typing import Type

from RS2.imageio.base_reader_writer import BaseReaderWriter
from RS2.imageio.simpleitk_reader_writer import SimpleITKIO


def get_reader_writer() -> Type[BaseReaderWriter]:
    ret = SimpleITKIO
    if ret is None:
        raise RuntimeError("Unable to find reader writer class. Please make sure this class is located in the "
                           "imageio module.")
    else:
        return ret
