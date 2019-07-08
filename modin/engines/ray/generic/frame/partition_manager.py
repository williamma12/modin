import ray
import numpy as np
from ray.worker import RayTaskError

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.engines.ray.utils import handle_ray_task_error


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
            if not isinstance(self._partitions_cache[0][0].length(), int):
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
                            if isinstance(obj.length(), int)
                            else ray.get(obj.length().oid)
                            for obj in self._partitions_cache.T[0]
                        ]
                    )
            else:
                self._lengths_cache = np.array(
                    [
                        obj.length()
                        if isinstance(obj.length(), int)
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
            if not isinstance(self._partitions_cache[0][0].width(), int):
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
                            if isinstance(obj.width(), int)
                            else ray.get(obj.width().oid)
                            for obj in self._partitions_cache[0]
                        ]
                    )
            else:
                self._widths_cache = np.array(
                    [
                        obj.width()
                        if isinstance(obj.width(), int)
                        else ray.get(obj.width().oid)
                        for obj in self._partitions_cache[0]
                    ]
                )
        return self._widths_cache

    def manual_shuffle(self, axis, shuffle_func, lengths):
        """Shuffle the partitions based on the `shuffle_func`.

        Args:
            axis: The axis to shuffle across.
            shuffle_func: The function to apply before splitting the result.
            lengths: The length of each partition to split the result into.

        Returns:
             A new BaseFrameManager object, the type of object that called this.
        """
        self_lengths = self.block_widths if axis else self.block_lengths
        other_cumsum = np.insert(0, 0, np.cumsum(lengths))
        
        # Calculate sharding of each block
        shard_data = []
        shard_partitions = []
        ind = 0
        self_ind = 0
        self_blk_ind = 0
        remainder = 0

        for other_len in lengths:
            temp_ind = 0
            split_lengths = []
            split_partitions = []
            while temp_ind < other_len:
                split_partitions.append(self_blk_ind)
                prev_ind = temp_ind
                self_blk_len = self_lengths[self_blk_ind]
                if remainder > 0:
                    temp_ind += remainder
                    remainder = 0
                else:
                    temp_ind += self_lengths[self_blk_ind]
                    self_blk_ind += 1
                if temp_ind > other_len:
                    remainder = temp_ind - other_len
                    temp_ind = other_len
                    self_ind -= 1
                new_blk_ind = prev_ind + (temp_ind - prev_ind)
                split_lengths.append((prev_ind, new_blk_ind))
                self_ind = new_blk_ind % self_blk_len
            shard_data.append(split_lengths)
            shard_partitions.append(split_partitions)

        # Generate new partitions
        result = []
        partitions = self.partitions if axis else self.partitions.T
        avail_actors = get_available_actors(99999)
        for repeat_idx in range(len(partitions)):
            axis_parts = []
            for shard_idx in range(len(shard_partitions)):
                partition_args = []
                for part_idx in shard_partitions[shard_idx]:
                    partition_args.append(partitions[repeat_idx][part_idx] if axis else partitions[part_idx][repeat_idx])
                if axis != actor_axis:
                    # Round robin
                    actor = avail_actors[repeat_idx % len(avail_actors)]
                else:
                    actor = actors[0][shard_idx] if axis else actors[shard_idx][0]
                part_data = actor.shuffle.remote(axis, shuffle_func, other_cumsum[shard_idx:shard_idx+2], shard_data[shard_idx], *partition_args)
                part_width = lengths[shard_idx] if axis else self.block_widths[repeat_idx]
                part_length = self.block_lengths[repeat_idx] if axis else lengths[shard_idx]
                axis_parts.append(self._partition_class(part_data, part_length, part_width))
            result.append(axis_parts)
        return self.__constructor__(result) if axis else self.__constructor__(result.T)
