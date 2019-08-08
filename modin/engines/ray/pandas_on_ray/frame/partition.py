from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
import ray
from ray.worker import RayTaskError

from modin.engines.base.frame.partition import BaseFramePartition
from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.ray.utils import handle_ray_task_error
from pandas.core.dtypes.common import is_list_like


class PandasOnRayFramePartition(BaseFramePartition):
    def __init__(self, object_id, length=None, width=None, call_queue=[]):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width

    def get(self):
        """Gets the object out of the plasma store.

        Returns:
            The object from the plasma store.
        """
        if len(self.call_queue):
            self.drain_call_queue()
        try:
            return ray.get(self.oid)
        except RayTaskError as e:
            handle_ray_task_error(e)

    def apply(self, func, actor, **kwargs):
        """Apply a function to the object stored in this partition.

        Note: It does not matter if func is callable or an ObjectID. Ray will
            handle it correctly either way. The keyword arguments are sent as a
            dictionary.

        Args:
            func: The function to apply.

        Returns:
            A RayRemotePartition object.
        """
        oid = self.oid
        call_queue = self.call_queue + [(func, kwargs)]
        # new_obj, result, length, width = deploy_ray_func.remote(call_queue, oid)
        new_obj, result, length, width = actor.run._remote(
                args=[
                    deploy_ray_func,
                    call_queue,
                    oid,
                    ],
                num_return_vals=4
                )
        if len(self.call_queue) > 0:
            self.oid = new_obj
            self.call_queue = []
        return PandasOnRayFramePartition(
            result, PandasOnRayFramePartition(length), PandasOnRayFramePartition(width)
        )

    def add_to_apply_calls(self, func, **kwargs):
        return PandasOnRayFramePartition(
            self.oid, call_queue=self.call_queue + [(func, kwargs)]
        )

    def drain_call_queue(self):
        if len(self.call_queue) == 0:
            return
        oid = self.oid
        call_queue = self.call_queue
        _, self.oid, length, width = deploy_ray_func_NO_ACTOR.remote(call_queue, oid)
        self.call_queue = []
        if self._length_cache is None:
            self._length_cache = PandasOnRayFramePartition(length)
        if self._width_cache is None:
            self._width_cache = PandasOnRayFramePartition(width)

    def split(self, actor, axis, is_transposed, splits):
        """Split partition along axis given list of resulting index groups (splits).

        Args:
            axis: Axis to split along.
            splits: List of list of indexes to group together.
            is_transposed: True if the partition needs to be is_transposed first.

        Returns:
            Returns PandasOnRayFramePartitions for each of the resulting splits.
        """
        # Check to make sure that if the split is just the original partition.
        if (
            len(splits) == 1 and is_list_like(splits[0])
            and len(splits[0]) == (self.width() if axis else self.length())
            and all(i == splits[0][i] for i in range(len(splits[0])))
        ):
            if is_transposed:
                call_queue = self.call_queue + [(pandas.DataFrame.transpose, {})]
            else:
                call_queue = self.call_queue
            return [
                PandasOnRayFramePartition(
                    self.oid, self._length_cache, self._width_cache, call_queue
                )
            ]
        else:
            if isinstance(splits[0], BaseFramePartition):
                splits = [split.oid for split in splits]
            # new_parts = deploy_ray_split._remote(
            #     args=[self.call_queue, self.oid, axis, is_transposed, *splits],
            #     num_return_vals=3 + len(splits),
            # )
            new_parts = actor.run._remote(
                args=[deploy_ray_split, self.call_queue, self.oid, axis, is_transposed, *splits],
                num_return_vals=3 + len(splits),
            )

            # Get and update self after draining call queue
            new_self_data, new_parts = new_parts[:3], new_parts[3:]
            self.oid = new_self_data[0]
            self._length_cache = PandasOnRayFramePartition(new_self_data[1])
            self._width_cache = PandasOnRayFramePartition(new_self_data[2])
            self.call_queue = []

            return [PandasOnRayFramePartition(new_part) for new_part in new_parts]

    @classmethod
    def shuffle(cls, actor, axis, func, on_partitions, partitions, other_on_partitions, other_partitions, length=None, width=None, fill_value=np.nan):
        """Takes the partitions combines them based on the indices.

        Args:
            axis: Axis to combine the partitions by.
            func: Function to apply after creating the new partition.
            n_other_partitions: Number of other partitions in partitions list.
            *partitions: List of partitions to combine.
            other_partition: Other partition to append to each partition
            along the other axis.
            length: Length of the resulting partition.
            width: Width of the resulting partition.

        Returns:
            A `BaseFramePartition` object.
        """
        on_oids = []
        for on_part in on_partitions:
            on_oids.append(on_part.oid)
        other_oids = []
        for other_part in other_partitions:
            other_oids.append(other_part.oid)
        other_on_oids = []
        for other_on_part in other_on_partitions:
            other_on_oids.append(other_on_part.oid)
        call_queues = []
        part_oids = []
        for part in partitions:
            if isinstance(part, PandasOnRayFramePartition):
                part_oids.append(part.oid)
                call_queues.append(part.call_queue)
            else:
                part_oids.append(part)
                call_queues.append(None)
        result, ray_length, ray_width = actor.run._remote(
                args=[
                    deploy_ray_shuffle,
                    axis,
                    func,
                    length if axis else width,
                    fill_value,
                    len(on_partitions),
                    len(other_on_partitions),
                    len(other_partitions),
                    call_queues,
                    *on_oids,
                    *other_on_oids,
                    *other_oids,
                    *part_oids
                    ],
                num_return_vals=3
                )
        length = length if func is None else PandasOnRayFramePartition(ray_length)
        width = width if func is None else PandasOnRayFramePartition(ray_width)
        return PandasOnRayFramePartition(result, length, width)

    def __copy__(self):
        return PandasOnRayFramePartition(
            self.oid, self._length_cache, self._width_cache
        )

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series
        return dataframe

    def to_numpy(self):
        """Convert the object stored in this parition to a Numpy Array.

        Returns:
            A Numpy Array.
        """
        return self.apply(lambda df: df.values).get()

    def mask(self, row_indices, col_indices):
        new_obj = self.add_to_apply_calls(
            lambda df: pandas.DataFrame(df.iloc[row_indices, col_indices])
        )
        new_obj._length_cache, new_obj._width_cache = len(row_indices), len(col_indices)
        return new_obj

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        return PandasOnRayFramePartition(ray.put(obj), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """Put a callable function into the plasma store for use in `apply`.

        Args:
            func: The function to preprocess.

        Returns:
            A ray.ObjectID.
        """
        return ray.put(func)

    @classmethod
    def length_extraction_fn(cls):
        return length_fn_pandas

    @classmethod
    def width_extraction_fn(cls):
        return width_fn_pandas

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())


@ray.remote(num_return_vals=4)
def deploy_ray_func_NO_ACTOR(call_queue, partition):  # pragma: no cover
    def deserialize(obj):
        if isinstance(obj, ray.ObjectID):
            return ray.get(obj)
        return obj

    if len(call_queue) > 1:
        for func, kwargs in call_queue[:-1]:
            func = deserialize(func)
            kwargs = deserialize(kwargs)
            try:
                partition = func(partition, **kwargs)
            except ValueError:
                partition = func(partition.copy(), **kwargs)
    func, kwargs = call_queue[-1]
    func = deserialize(func)
    kwargs = deserialize(kwargs)
    try:
        result = func(partition, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
    # don't want the error to propagate to the user, and we want to avoid copying unless
    # we absolutely have to.
    except ValueError:
        result = func(partition.copy(), **kwargs)
    return (
        partition if len(call_queue) > 1 else None,
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
    )


def deploy_ray_func(call_queue, partition):  # pragma: no cover
    def deserialize(obj):
        if isinstance(obj, ray.ObjectID):
            return ray.get(obj)
        return obj

    if len(call_queue) > 1:
        for func, kwargs in call_queue[:-1]:
            func = deserialize(func)
            kwargs = deserialize(kwargs)
            try:
                partition = func(partition, **kwargs)
            except ValueError:
                partition = func(partition.copy(), **kwargs)
    func, kwargs = call_queue[-1]
    func = deserialize(func)
    kwargs = deserialize(kwargs)
    try:
        result = func(partition, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
    # don't want the error to propagate to the user, and we want to avoid copying unless
    # we absolutely have to.
    except ValueError:
        result = func(partition.copy(), **kwargs)
    return (
        partition if len(call_queue) > 1 else None,
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
    )


def deploy_ray_split(
    call_queue, partition, axis, is_transposed, *splits
):  # pragma: no cover
    def deserialize(obj):
        if isinstance(obj, ray.ObjectID):
            return ray.get(obj)
        return obj

    # Drain call queue.
    if len(call_queue) > 0:
        for func, kwargs in call_queue:
            func = deserialize(func)
            kwargs = deserialize(kwargs)
            try:
                partition = func(partition, **kwargs)
            except ValueError:
                partition = func(partition.copy(), **kwargs)

    # Orient and cut up partition.
    part = partition.T if is_transposed else partition

    # TODO: Workaround until pandas 0.25
    result = []
    for index in splits:
        if isinstance(index, np.ndarray):
            index = index.tolist()
        result.append(part.iloc[:, index] if axis else part.iloc[index])
    # result = [part.iloc[:, index] if axis else part.iloc[index] for index in splits]
    return [
        partition,
        len(partition) if hasattr(partition, "__len__") else 0,
        len(partition.columns) if hasattr(partition, "columns") else 0,
    ] + result


def deploy_ray_shuffle(
    axis, shuffle_func, length, fill_value, n_on_partitions, n_other_on_partitions, n_other_partitions, call_queues, *partitions
):  # pragma: no cover
    def deserialize(obj):
        if isinstance(obj, ray.ObjectID):
            return ray.get(obj)
        return obj

    # If no partitions, return empty dataframe.
    if len(partitions) == 0:
        return pandas.DataFrame(), 0, 0

    # Separate other_partitions and partitions.
    on_partitions = partitions[:n_on_partitions]
    other_on_partitions = partitions[n_on_partitions:n_on_partitions+n_other_on_partitions]
    other_partitions = partitions[n_on_partitions+n_other_on_partitions:n_on_partitions+n_other_on_partitions+n_other_partitions]
    partitions = partitions[n_on_partitions+n_other_on_partitions+n_other_partitions:]

    # Create partition from partitions.
    df_parts = []
    for i in range(len(partitions)):
        partition = partitions[i]
        call_queue = call_queues[i]
        if isinstance(partition, int):
            # Create empty partition. This is only reached during reindex and
            # not during sorts or joins where length argument is None.
            nan_len = partition
            df_part = pandas.DataFrame(
                np.repeat(fill_value, nan_len * length).reshape(
                    (length, nan_len) if axis else (nan_len, length)
                )
            )
        else:
            if partition.empty:
                continue
            # Drain call queue.
            if len(call_queue) > 0:
                for func, kwargs in call_queue:
                    func = deserialize(func)
                    kwargs = deserialize(kwargs)
                    try:
                        partition = func(partition, **kwargs)
                    except ValueError:
                        partition = func(partition.copy(), **kwargs)
            df_part = partition

        # Reset index and columns for consistent concat behavior.
        df_part.index = pandas.RangeIndex(len(df_part))
        df_part.columns = pandas.RangeIndex(len(df_part.columns))
        df_parts.append(df_part)

    # Return if no parts to append.
    if len(df_parts) == 0:
        return pandas.DataFrame(), 0, 0

    df = pandas.concat(df_parts, axis=axis)

    # Add on partitions to append.
    if n_on_partitions > 0:
        for on_part in on_partitions:
            if axis:
                on_part.columns = pandas.RangeIndex(len(on_part.columns))
            else:
                on_part.index = pandas.RangeIndex(len(on_part))
        on_part = pandas.concat(on_partitions, axis=axis)
        df = pandas.concat([df, on_part], axis=axis^1)

    # Make sure internal indices are correct.
    if axis:
        df.columns = pandas.RangeIndex(len(df.columns))
    else:
        df.index = pandas.RangeIndex(len(df))

    if n_other_partitions > 0:
        for other_part in other_partitions:
            if axis:
                other_part.columns = pandas.RangeIndex(len(other_part.columns))
            else:
                other_part.index = pandas.RangeIndex(len(other_part))
        other_partition = pandas.concat(other_partitions, axis=axis)
        for other_on_part in other_on_partitions:
            if axis:
                other_on_part.columns = pandas.RangeIndex(len(other_on_part.columns))
            else:
                other_on_part.index = pandas.RangeIndex(len(other_on_part))
        other_on_part = pandas.concat(other_on_partitions, axis=axis)
        if axis:
            other_on_part.columns = pandas.RangeIndex(len(other_on_part.columns))
        else:
            other_on_part.index = pandas.RangeIndex(len(other_on_part))
        other_partition = pandas.concat([other_partition, other_on_part], axis=axis^1)
    else:
        other_partition = None

    # Apply post-shuffle function.
    if shuffle_func is not None:
        result = shuffle_func(df, other=other_partition)
    else:
        result = df

    return (
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
    )
