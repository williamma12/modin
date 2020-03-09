import numpy as np

from .base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.data_management.functions import (
    FoldFunction,
    MapFunction,
    MapReduceFunction,
    ReductionFunction,
)

def _get_axis(axis):
    if axis == 0:
        return lambda self: self._modin_frame.index
    else:
        return lambda self: self._modin_frame.columns


def _set_axis(axis):
    if axis == 0:

        def set_axis(self, idx):
            self._modin_frame.index = idx

    else:

        def set_axis(self, cols):
            self._modin_frame.columns = cols

    return set_axis


class GenericQueryCompiler(BaseQueryCompiler):
    """Development query compiler to have different backends simultaneously"""

    def __init__(self, modin_frame):
        self._modin_frame = modin_frame

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    isna = MapFunction.register("isna", dtypes=np.bool)

    # Head/Tail/Front/Back
    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the first n rows of the original QueryCompiler.
        """
        return self.__constructor__(self._modin_frame.head(n))

    def tail(self, n):
        """Returns the last n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the last n rows of the original QueryCompiler.
        """
        return self.__constructor__(self._modin_frame.tail(n))

    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the first n columns of the original QueryCompiler.
        """
        return self.__constructor__(self._modin_frame.front(n))

    def back(self, n):
        """Returns the last n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the last n columns of the original QueryCompiler.
        """
        return self.__constructor__(self._modin_frame.back(n))

    # End Head/Tail/Front/Back

