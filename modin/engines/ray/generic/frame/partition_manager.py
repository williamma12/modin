import ray
import numpy as np
import pandas
from ray.worker import RayTaskError

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.engines.ray.utils import handle_ray_task_error
from modin.data_management.utils import compute_partition_shuffle
from pandas.api.types import is_integer


class RayFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    def __init__(self, partitions):
        self.partitions = partitions

    # We override these for performance reasons.
    # Lengths of the blocks
    _lengths_cache = None
    # Widths of the blocks
    _widths_cache = None

    # These are set up as properties so that we only use them when we need
    # them. We also do not want to trigger this computation on object creation.
    @property
    def block_lengths(self):
        """Gets the lengths of the blocks.

        Note: This works with the property structure `_lengths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._lengths_cache is None:
            if not is_integer(self._partitions_cache[0][0].length()):
                try:
                    # The first column will have the correct lengths. We have an
                    # invariant that requires that all blocks be the same length in a
                    # row of blocks.
                    self._lengths_cache = np.array(
                        ray.get(
                            [obj.length().oid for obj in self._partitions_cache.T[0]]
                        )
                        if len(self._partitions_cache.T) > 0
                        else []
                    )
                except RayTaskError as e:
                    handle_ray_task_error(e)
                except AttributeError:
                    self._lengths_cache = np.array(
                        [
                            obj.length()
                            if is_integer(obj.length())
                            else ray.get(obj.length().oid)
                            for obj in self._partitions_cache.T[0]
                        ]
                    )
            else:
                self._lengths_cache = np.array(
                    [
                        obj.length()
                        if is_integer(obj.length())
                        else ray.get(obj.length().oid)
                        for obj in self._partitions_cache.T[0]
                    ]
                )
        return self._lengths_cache

    @property
    def block_widths(self):
        """Gets the widths of the blocks.

        Note: This works with the property structure `_widths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._widths_cache is None:
            if not is_integer(self._partitions_cache[0][0].width()):
                try:
                    # The first column will have the correct lengths. We have an
                    # invariant that requires that all blocks be the same width in a
                    # column of blocks.
                    self._widths_cache = np.array(
                        ray.get([obj.width().oid for obj in self._partitions_cache[0]])
                        if len(self._partitions_cache) > 0
                        else []
                    )
                except RayTaskError as e:
                    handle_ray_task_error(e)
                except AttributeError:
                    self._widths_cache = np.array(
                        [
                            obj.width()
                            if is_integer(obj.width())
                            else ray.get(obj.width().oid)
                            for obj in self._partitions_cache[0]
                        ]
                    )
            else:
                self._widths_cache = np.array(
                    [
                        obj.width()
                        if is_integer(obj.width())
                        else ray.get(obj.width().oid)
                        for obj in self._partitions_cache[0]
                    ]
                )
        return self._widths_cache

    def manual_shuffle(self, axis, shuffle_func, lengths, transposed=False):
        """Shuffle the partitions based on the `shuffle_func`.

        Args:
            axis: The axis to shuffle across.
            shuffle_func: The function to apply before splitting the result.
            lengths: The length of each partition to split the result into.

        Returns:
             A new BaseFrameManager object, the type of object that called this.
        """

        @ray.remote
        class ShuffleActors(object):
            def shuffle(
                self, axis, func, internal_indices, transposed, indices, *partitions
            ):
                if len(indices) == 0:
                    return pandas.DataFrame()
                df_parts = []
                for i, part_indices in enumerate(indices):
                    partition = partitions[i].T if transposed else partitions[i]
                    start, end = part_indices
                    df_parts.append(
                        partition.iloc[:, start:end]
                        if axis
                        else partition.iloc[start:end]
                    )
                df = pandas.concat(df_parts, axis=axis)
                result = func(df, internal_indices)
                return result

        partition_shuffle = compute_partition_shuffle(
            self.block_widths if axis else self.block_lengths, lengths
        )
        internal_indices = np.insert(np.cumsum(lengths), 0, 0)

        result = []
        partitions = self.partitions if axis else self.partitions.T
        # We create one actor for each partition in the result
        actors = [ShuffleActors.remote() for _ in range(len(lengths) * len(partitions))]
        for row_idx in range(len(partitions)):
            axis_parts = []
            for col_idx in range(len(lengths)):
                # Compile the arguments needed for shuffling
                partition_args = []
                indices = []
                for part_idx, index in partition_shuffle[col_idx]:
                    partition_args.append(
                        partitions[row_idx][part_idx].oid
                        if axis
                        else partitions[part_idx][row_idx].oid
                    )
                    indices.append(index)
                actor = actors[col_idx + row_idx * len(lengths)]

                # Create shuffled data and create partition
                part_data = actor.shuffle.remote(
                    axis,
                    shuffle_func,
                    internal_indices[col_idx : col_idx + 2],
                    transposed,
                    indices,
                    *partition_args
                )
                part_width = lengths[col_idx] if axis else self.block_widths[row_idx]
                part_length = self.block_lengths[row_idx] if axis else lengths[col_idx]
                axis_parts.append(
                    self._partition_class(part_data, part_length, part_width)
                )
            result.append(axis_parts)
        return (
            self.__constructor__(np.array(result))
            if axis
            else self.__constructor__(np.array(result).T)
        )
