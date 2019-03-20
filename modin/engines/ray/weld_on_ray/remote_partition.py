from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray

from modin.engines.base.remote_partition import BaseRemotePartition
from modin.data_management.utils import length_fn_weld, width_fn_weld


class WeldOnRayRemotePartition(BaseRemotePartition):
    """This method implements the interface in `BaseRemotePartitions`."""

    def __init__(self, object_id):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.call_queue = []

    def get(self):
        """Gets the object out of the plasma store.

        Returns:
            The object from the plasma store.
        """
        # This might stay the same. Make this output a pandas DataFrame.
        if len(self.call_queue):
            return self.apply(lambda x: x).get()

        return ray.get(self.oid)

    def apply(self, func, **kwargs):
        """Apply a function to the object stored in this partition.

        Note: It does not matter if func is callable or an ObjectID. Ray will
            handle it correctly either way. The keyword arguments are sent as a
            dictionary.

        Args:
            func: The function to apply.

        Returns:
            A WeldOnRayRemotePartition object.
        """
        oid = self.oid
        self.call_queue.append((func, kwargs))

        def call_queue_closure(oid_obj, call_queues):

            for func, kwargs in call_queues:
                if isinstance(func, ray.ObjectID):
                    func = ray.get(func)
                if isinstance(kwargs, ray.ObjectID):
                    kwargs = ray.get(kwargs)

                # This is where the function is applied.
                # Note: Some of the functions call pandas.DataFrame.<method> here.
                # Note: Some are also lambdas e.g. lambda df: df.<method>.
                oid_obj = func(oid_obj, **kwargs)

            return oid_obj

        oid = deploy_ray_func.remote(
            call_queue_closure, oid, kwargs={"call_queues": self.call_queue}
        )
        self.call_queue = []

        return WeldOnRayRemotePartition(oid)

    def add_to_apply_calls(self, func, **kwargs):
        # This is for lazy operators that don't change the state (e.g. iloc)
        self.call_queue.append((func, kwargs))
        return self

    def __copy__(self):
        return WeldOnRayRemotePartition(object_id=self.oid)

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        # This will likely stay the same
        dataframe = self.get().to_pandas()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        import grizzly.grizzly as gr
        if isinstance(obj, gr.DataFrameWeld):
            grizzly_obj = obj
        else:
            grizzly_obj = gr.DataFrameWeld(obj)
        return WeldOnRayRemotePartition(ray.put(grizzly_obj))

    @classmethod
    def preprocess_func(cls, func):
        """Put a callable function into the plasma store for use in `apply`.

        Args:
            func: The function to preprocess.

        Returns:
            A ray.ObjectID.
        """
        # This is where we serialize functions that will be applied.
        # There may be no need for this.
        return ray.put(func)

    @classmethod
    def length_extraction_fn(cls):
        return length_fn_weld

    @classmethod
    def width_extraction_fn(cls):
        return width_fn_weld

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())


@ray.remote
def deploy_ray_func(func, partition, kwargs):
    """Deploy a function to a partition in Ray.

    Args:
        func: The function to apply.
        partition: The partition to apply the function to.
        kwargs: A dictionary of keyword arguments for the function.

    Returns:
        The result of the function.
    """
    # This is where functiona are applied remotely. If we are just using Ray to schedule
    # Grizzly, there is a good chance this isn't necessary.
    try:
        return func(partition, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate
    # on it. We don't want the error to propagate to the user, and we want to
    # avoid copying unless we absolutely have to.
    except ValueError:
        return func(partition.copy(), **kwargs)
