from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import pandas
import numpy as np
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
    # @profile
    def sort_split(
        cls,
        partitions,
        is_transposed,
        on_indices,
        na_position,
        bins=[],
        bin_boundaries=[],
    ):
        """Split partition along axis given list of resulting index groups (splits).

        Note: Naming convention for column or row follows if axis is 1.

        Args:
            partitions: Dictionary of original row index to row partitions.
            is_transposed: True if the partition needs to be is_transposed first.
            on_indices: Dictionary mapping of partition_row_index to 
            ([internal_indices], [on_orders]).
            na_position: "first" puts NaNs first, "last" puts NaNs last.
            bins: Dataframe containing bins to sort the values by.
            bin_boundaries: Dictionary of (partitions_column_index, internal_index) of 
            the bin boundary values.

        Returns:
            Returns PandasOnRayFramePartitions for each of the resulting splits.
        """
        # Create actors and get the on partitions and bins, if needed.
        sort_split_actors = defaultdict(list)
        on_parts = []
        bin_parts = []
        for row_idx, row_parts in partitions.items():
            on_row_parts = []
            on_row_indices, on_row_ordering = on_indices[row_idx]
            for col_idx, block in enumerate(row_parts):
                # Create and save actor
                actor = SortSplitActor._remote(
                    [cls.axis, is_transposed, block.call_queue, block.oid, on_indices], num_cpus=1/(len(partitions)*len(row_parts))
                )
                sort_split_actors[row_idx].append(actor)

                # Get the on partition and bins, if needed.
                if (
                    len(bin_boundaries) > 0
                    and col_idx in bin_boundaries
                    and 0 in on_row_ordering
                ):
                    on_partition, bin_boundary = actor.get_on_partitions_and_bins._remote(
                        [on_row_indices, on_row_ordering, bin_boundaries[col_idx]],
                        num_return_vals=2,
                    )
                    on_row_parts.append(on_partition)
                    bin_parts.append(bin_boundary)
                else:
                    on_partition = actor.get_on_partitions_and_bins.remote(
                        on_row_indices, on_row_ordering
                    )
                    on_row_parts.append(on_partition)
            on_parts.append(on_row_parts)

        # for row in on_parts:
        #     ray.get(row)
        # ray.get(bin_parts)

        # Merge on parts to line up with the cls.axis.
        n_bins = (
            sum([len(internal_index) for internal_index in bin_boundaries.values()]) + 1
        )
        on_partitions = []
        splits = []
        on_parts = np.array(on_parts)
        _, on_parts_columns = on_parts.shape
        for col_idx in range(on_parts_columns):
            on_partitions_and_split = concat_partitions_and_compute_splits._remote(
                [cls.axis, na_position, bin_parts, *on_parts[:, col_idx]],
                num_return_vals=2*n_bins,
            )
            on_partitions.append(on_partitions_and_split[:n_bins])
            splits.append(on_partitions_and_split[n_bins:])

        # for row in on_partitions:
        #     ray.get(row)
        # for split in splits:
        #     ray.get(split)

        # Append the on partitions here to avoid doing it multiple times when shuffling.
        if n_bins > 1:
            on_partitions = np.array(on_partitions)
            results = []
            for col in range(n_bins):
                if len(on_partitions) > 1:
                    result = concat_partitions_and_compute_splits.remote(cls.axis, None, None, *on_partitions[:, col])
                else:
                    result = np.squeeze(on_partitions[:, col])
                results.append(cls.partition_type(result))
            on_partitions = results
        else:
            on_partitions = [cls.partition_type(on_partitions[0][0])]

        # Send on partitions to remaining actors to get the splits and splits for old on partitions.
        on_old_partitions = {
            row_idx: [
                [cls.partition_type(oid)
                for oid in sort_split_actors[row_idx][col_idx].split_partitions._remote(
                    args=[*splits[col_idx]], num_return_vals=max(2, len(splits[col_idx]))
                )]
                for col_idx in range(on_parts_columns)
            ]
            for row_idx in sort_split_actors.keys()
        }

        # for row in on_old_partitions.values():
        #     for blocks in row:
        #         for block in blocks:
        #             ray.get(block.oid)

        return bins, splits, on_old_partitions, on_partitions

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
class SortSplitActor(object):  # pragma: no cover
    @ray.method(num_return_vals=0)
    def __init__(self, axis, is_transposed, call_queue, partition, on_indices):
        self.axis = axis
        self.on_indices = on_indices

        def deserialize(obj):
            if isinstance(obj, ray.ObjectID):
                return ray.get(obj)
            return obj

        # Drain call queues
        if len(call_queue) > 0:
            for func, kwargs in call_queue:
                func = deserialize(func)
                kwargs = deserialize(kwargs)
                try:
                    partition = func(partition, **kwargs)
                except ValueError:
                    partition = func(partition.copy(), **kwargs)

        self.partition = partition.T if is_transposed else partition

    def get_on_partitions_and_bins(self, on_indices, on_orders, bin_boundaries=[]):
        """Get the on partitions and bins, if needed.

        Args:
            on_indices: List of internal indices of the on rows in this partition.
            on_orders: List of the sort order of the corresponding on_indices.
            bin_boundaries: Bin boundary column index value. Empty list of no bin
            boundaries are needed.
        
        Returns:
            On partition and, if needed, bins partition.
        """
        # Get and set up on labels correctly.
        on_partition = (
            self.partition.iloc[on_indices]
            if self.axis
            else self.partition.iloc[:, on_indices]
        )
        on_labels = ["__sort_{}__".format(i) for i in on_orders]
        on_partition.index = (
            on_labels if self.axis else pandas.RangeIndex(len(on_partition))
        )
        on_partition.columns = (
            pandas.RangeIndex(len(on_partition.columns)) if self.axis else on_labels
        )

        # Get bin boundaries, if necessary.
        if len(bin_boundaries) == 0:
            return on_partition
        else:
            # TODO: if bin val is none, pick new one.
            bins = (
                on_partition.loc["__sort_0__", bin_boundaries].values
                if self.axis
                else on_partition.loc[bin_boundaries, "__sort_0__"].values
            )
            return on_partition, bins

    def split_partitions(self, *splits):
        import time
        start = time.time()
        if len(splits) == 1:
            result = self.partition.reindex(splits[0], axis=self.axis), None
        else:
            result = [
                pandas.DataFrame()
                if len(index) == 0
                else self.partition.iloc[:, index]
                if self.axis
                else self.partition.iloc[index]
                for index in splits
            ]
        # print(time.time() - start)
        return result


@ray.remote
def concat_partitions_and_compute_splits(
    axis, na_position="last", bins=None, *partitions
):
    """Concats partitions to one dataframe and computs splits, if necessary.

    Args:
        axis: Axis to merge by for each row of the dataframe.
        partitions: List of dataframes.
        bins: Bins to calculate the splits with.

    Returns:
        One dataframe that is partitions concated on axis.
    """
    result = pandas.concat(partitions, axis=axis)
    if axis == 0:
        result = result.reset_index(drop=True)
    else:
        result.columns = pandas.RangeIndex(len(result.columns))
    if bins is None:
        return result
    elif len(bins) == 0:
        return result, list()
    else:
        df = result if axis else result.T
        df = df.loc["__sort_0__"]

        bins = np.sort(np.concatenate(ray.get(bins)))
        result = []
        splits = []
        if na_position is "first":
            array = df.fillna(df.min()).values
        else:
            array = df.fillna(df.max()).values
        bin_idx = np.searchsorted(bins, array)
        for i in range(len(bins)+1):
            # if i == len(bins) - 1:
            #     indices = np.argwhere((bin_idx == i) | (bin_idx == i+1)).flatten()
            # else:
            #     indices = np.argwhere(bin_idx == i).flatten()
            indices = np.argwhere(bin_idx == i).flatten()
            splits.append(indices)
            result.append(df[indices] if axis else df[indices].T)
        return result + splits

        # bins = np.concatenate(ray.get(bins))
        # df = result if axis else result.T
        # array = df.loc["__sort_0__"]
        # result = []
        # splits = []
        # na_first = na_position is "first"
        # indices = np.arange(len(array))
        # if hasattr(bins[0], "__sub__"):
        #     bins = np.insert(bins, 0, df.loc["__sort_0__"].min())
        #     bins = np.append(bins, df.loc["__sort_0__"].max())
        #     bins = pandas.Series(np.sort(bins))
        #     result = []
        #     na_first = na_position is "first"
        #     indices = np.arange(len(array))
        #     splits = pandas.cut(df.loc["__sort_0__"], bins, right=True, labels=False, include_lowest=True)
        #     if na_position is "first":
        #         splits.fillna(0)
        #     else:
        #         splits = splits.fillna(len(bins.index)-1)
        #     result_splits = []
        #     result_parts = []
        #     for i in range(len(bins.index)-1):
        #         indices = splits.index[splits == i]
        #         result_splits.append(list(indices))
        #         result_parts.append(df[indices] if axis else df[indices].T)
        #     return result_parts + result_splits
        # else:
        #     bins = np.sort(bins)
        #     indices = np.arange(len(array))
        #     for bin_val in bins:
        #         if na_first:
        #             mask = (array <= bin_val) | (
        #                 array == np.NaN
        #             ) | (array is None)
        #             na_first = False
        #         else:
        #             mask = array <= bin_val
        #         index = indices[np.argwhere(mask).flatten()]
        #         splits.append(index)
        #         result.append(df[index] if axis else df[index].T)
        #         array = array[~mask]
        #         indices = indices[~mask]
        #         df = df.drop(index, axis=1)
        #     splits.append(indices)
        #     result.append(df if axis else df.T)
        #     return result + splits
