from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnRayFramePartition


class PandasOnRayFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    partition_type = PandasOnRayFramePartition
    instance_type = ray.ObjectID

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        return deploy_ray_func._remote(
            args=(
                PandasFrameAxisPartition.deploy_axis_func,
                axis,
                func,
                num_splits,
                kwargs,
                maintain_partitioning,
            )
            + tuple(partitions),
            num_return_vals=num_splits * 3,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, kwargs, *partitions
    ):
        return deploy_ray_func._remote(
            args=(
                PandasFrameAxisPartition.deploy_func_between_two_axis_partitions,
                axis,
                func,
                num_splits,
                len_of_left,
                kwargs,
            )
            + tuple(partitions),
            num_return_vals=num_splits * 3,
        )

    @classmethod
    def split(cls, partitions, is_transposed, splits=[], sort=False, sort_on=None, sort_bins=[], sort_distribute_on=False):
        """Split partition along axis given list of resulting index groups (splits).

        Args:
            is_transposed: True if the partition needs to be is_transposed first.
            splits: List of list of indexes to group together.
            sort: True if we should sort the values.
            sort_on: Label to sort the values by.
            sort_bins: Bins to sort the values by.
            sort_distribute_on: True, then the on column(s) will be propragated to 
            the all partitions. Useful for joins and groupbys.

        Returns:
            Returns PandasOnRayFramePartitions for each of the resulting splits.
        """
        new_parts = deploy_ray_split._remote(
            args=[self.call_queue, self.oid, axis, splits, is_transposed, sort],
            num_return_vals=3 + len(splits),
        )

        # Get and update self after draining call queue
        new_self_data, new_parts = new_parts[:3], new_parts[3:]
        self.oid = new_self_data[0]
        self._length_cache = PandasOnRayFramePartition(new_self_data[1])
        self._width_cache = PandasOnRayFramePartition(new_self_data[2])
        self.call_queue = []

        return [PandasOnRayFramePartition(new_part) for new_part in new_parts]

    def _wrap_partitions(self, partitions):
        return [
            self.partition_type(
                partitions[i],
                self.partition_type(partitions[i + 1]),
                self.partition_type(partitions[i + 2]),
            )
            for i in range(0, len(partitions), 3)
        ]


class PandasOnRayFrameColumnPartition(PandasOnRayFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnRayFrameRowPartition(PandasOnRayFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1


@ray.remote
def deploy_ray_func(func, *args):  # pragma: no cover
    """Run a function on a remote partition.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)

    Args:
        func: The function to run.

    Returns:
        The result of the function `func`.
    """
    result = func(*args)
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns)
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns)]]
    else:
        return [i for r in result for i in [r, None, None]]


@ray.remote
class SplitActor(object):
    def __init__(self, axis, is_transposed, call_queues, partitions, splits, sort, sort_bins, sort_distribute_on):
        self.axis = axis
        self.is_transposed = is_transposed
        self.call_queues = call_queues
        self.partitions = partitions
        self.sort = sort
        self.sort_on = sort_on if is_list_like(sort_on) else [sort_on]
        self.sort_bins = sort_bins
        self.sort_distribute_on = sort_distribute_on

    def deploy_ray_split(self):
        if self.splits or (self.sort and len(self.sort_bins) == self.sort_n_bins):
            if self.sort:
                self.sort_bins.sort()
            return self._split_partitions(axis, transposed, call_queues, partitions)
        else:
            # Get bin value from self and send it to everyone
            return False

    def add_bin(self, bin_value):
        self.sort_bins.append(bin_value)
        self.deploy_ray_split()

    def _split_partitions(self):  # pragma: no cover
        def deserialize(obj):
            if isinstance(obj, ray.ObjectID):
                return ray.get(obj)
            return obj

        partition = self.partition
        # Drain call queue.
        if len(self.call_queue) > 0:
            for func, kwargs in self.call_queue:
                func = deserialize(func)
                kwargs = deserialize(kwargs)
                try:
                    partition = func(partition, **kwargs)
                except ValueError:
                    partition = func(partition.copy(), **kwargs)

        # Orient and cut up partition.
        part = partition.T if axis^1^is_transposed else partition
        if sort:
            df = part[self.sort_on]
            splits = []
            result = []
            for edge in self.sort_bins:
                if len(df) == 0:
                    break
                mask = df[self.sort_on[0]] <= edge
                indices = df.index[mask].to_list()
                splits.append(indices)
                result.append(df[mask])
                df = df[~mask]
        else:
            result = [None if len(index) == 0 else part[index] for index in self.splits]
        self.result = [
            partition,
            len(partition) if hasattr(partition, "__len__") else 0,
            len(partition.columns) if hasattr(partition, "columns") else 0,
        ] + result
        return True if self.sort else self.result
