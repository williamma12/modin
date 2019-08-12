from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas

from modin.error_message import ErrorMessage
from modin.data_management.utils import (
    compute_chunksize,
    _get_nan_block_id,
    set_indices_for_pandas_concat,
)


class BaseFrameManager(object):
    """Abstract Class that manages a set of `BaseFramePartition` objects, and
        structures them into a 2D numpy array. This object will interact with
        each of these objects through the `BaseFramePartition` API.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.
    def __init__(self, partitions, block_lengths=None, block_widths=None):
        """Init must accept a parameter `partitions` that is a 2D numpy array
            of type `_partition_class` (defined below). This method will be
            called from a factory.

        Args:
            partitions: A 2D numpy array of the type defined in
                `_partition_class`.
        """
        raise NotImplementedError("Must be implemented in children classes")

    @property
    def __constructor__(self):
        """Convenience method for creating new objects.

        Note: This is used by the abstract class to ensure the return type is the same
            as the child subclassing it.
        """
        return type(self)

    # Partition class is the class to use for storing each partition. It must
    # extend the `BaseFramePartition` class.
    _partition_class = None
    # Column partitions class is the class to use to create the column partitions.
    _column_partitions_class = None

    @classmethod
    def preprocess_func(cls, map_func):
        """Preprocess a function to be applied to `BaseFramePartition` objects.

        Note: If your `BaseFramePartition` objects assume that a function provided
            is serialized or wrapped or in some other format, this is the place
            to add that logic. It is possible that this can also just return
            `map_func` if the `apply` method of the `BaseFramePartition` object
            you are using does not require any modification to a given
            function.

        Args:
            map_func: The function to be preprocessed.

        Returns
            The preprocessed version of the `map_func` provided. Note: This
            does not require any specific format, only that the
            `BaseFramePartition.apply` method will recognize it (For the subclass
            being used).
        """
        return cls._partition_class.preprocess_func(map_func)

    # END Abstract Methods

    @classmethod
    def column_partitions(cls, partitions):
        """A list of `BaseFrameAxisPartition` objects.

        Note: Each value in this list will be an `BaseFrameAxisPartition` object.
            `BaseFrameAxisPartition` is located in `axis_partition.py`.

        Returns a list of `BaseFrameAxisPartition` objects.
        """
        return [cls._column_partitions_class(col) for col in partitions.T]

    @classmethod
    def row_partitions(cls, partitions):
        """A list of `BaseFrameAxisPartition` objects, represents column partitions.

        Note: Each value in this list will an `BaseFrameAxisPartition` object.
            `BaseFrameAxisPartition` is located in `axis_partition.py`.

        Returns a list of `BaseFrameAxisPartition` objects.
        """
        return [cls._row_partition_class(row) for row in partitions]

    # Lengths of the blocks
    _lengths_cache = None

    # These are set up as properties so that we only use them when we need
    # them. We also do not want to trigger this computation on object creation.
    @property
    def block_lengths(self):
        """Gets the lengths of the blocks.

        Note: This works with the property structure `_lengths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._lengths_cache is None:
            # The first column will have the correct lengths. We have an
            # invariant that requires that all blocks be the same length in a
            # row of blocks.
            self._lengths_cache = np.array(
                [obj.length().get() for obj in self._partitions_cache.T[0]]
                if len(self._partitions_cache.T) > 0
                else []
            )
        return self._lengths_cache

    # Widths of the blocks
    _widths_cache = None

    @property
    def block_widths(self):
        """Gets the widths of the blocks.

        Note: This works with the property structure `_widths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._widths_cache is None:
            # The first column will have the correct lengths. We have an
            # invariant that requires that all blocks be the same width in a
            # column of blocks.
            self._widths_cache = np.array(
                [obj.width().get() for obj in self._partitions_cache[0]]
                if len(self._partitions_cache) > 0
                else []
            )
        return self._widths_cache

    def _remove_empty_blocks(self):
        if self._widths_cache is not None:
            self._widths_cache = [width for width in self._widths_cache if width != 0]
        if self._lengths_cache is not None:
            self._lengths_cache = np.array(
                [length for length in self._lengths_cache if length != 0]
            )

    @property
    def shape(self):
        return int(np.sum(self.block_lengths)), int(np.sum(self.block_widths))

    def full_reduce(self, map_func, reduce_func, axis):
        """Perform a full reduce on the data.

        Note: This follows the 2-phase reduce paradigm, where each partition
            performs a local reduction (map_func), then partitions are brought
            together and the final reduction occurs.
        Args:
            map_func: The function that will be performed on all partitions.
                This is the local reduction on each partition.
            reduce_func: The final reduction function. This can differ from the
                `map_func`
            axis: The axis to perform this operation along
                (0 - index, 1 - columns)
        Returns:
            A Pandas Series
        """
        raise NotImplementedError("Blocked on Distributed Series")

    def groupby_reduce(self, axis, by, map_func, reduce_func):
        by_parts = np.squeeze(by.partitions)
        if len(by_parts.shape) == 0:
            by_parts = np.array([by_parts.item()])
        [obj.drain_call_queue() for obj in by_parts]
        new_partitions = self.__constructor__(
            np.array(
                [
                    [
                        part.apply(
                            map_func,
                            other=by_parts[col_idx].get()
                            if axis
                            else by_parts[row_idx].get(),
                        )
                        for col_idx, part in enumerate(self.partitions[row_idx])
                    ]
                    for row_idx in range(len(self.partitions))
                ]
            )
        )
        return new_partitions.map_across_full_axis(axis, reduce_func)

    @classmethod
    def map_across_blocks(cls, partitions, map_func):
        """Applies `map_func` to every partition.

        Args:
            map_func: The function to apply.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        new_partitions = np.array(
            [
                [part.apply(preprocessed_map_func) for part in row_of_parts]
                for row_of_parts in partitions
            ]
        )
        return new_partitions

    def lazy_map_across_blocks(self, map_func, kwargs):
        preprocessed_map_func = self.preprocess_func(map_func)
        new_partitions = np.array(
            [
                [
                    part.add_to_apply_calls(preprocessed_map_func, kwargs)
                    for part in row_of_parts
                ]
                for row_of_parts in self.partitions
            ]
        )
        return self.__constructor__(new_partitions)

    def copartition_datasets(
        self, axis, other, left_func, right_func, other_is_transposed
    ):
        """Copartition two BlockPartitions objects.

        Args:
            axis: The axis to copartition.
            other: The other BlockPartitions object to copartition with.
            left_func: The function to apply to left. If None, just use the dimension
                of self (based on axis).
            right_func: The function to apply to right. If None, check the dimensions of
                other and use the identity function if splitting needs to happen.

        Returns:
            A tuple of BlockPartitions objects, left and right.
        """
        if left_func is None:
            new_self = self
        else:
            new_self = self.map_across_full_axis(axis, left_func)

        # This block of code will only shuffle if absolutely necessary. If we do need to
        # shuffle, we use the identity function and then reshuffle.
        if right_func is None:
            if axis == 0 and not np.array_equal(
                other.block_lengths, new_self.block_lengths
            ):
                new_other = other.manual_shuffle(
                    axis,
                    lambda x: x,
                    new_self.block_lengths,
                    transposed=other_is_transposed,
                )
            elif axis == 1 and not np.array_equal(
                other.block_widths, new_self.block_widths
            ):
                new_other = other.manual_shuffle(
                    axis,
                    lambda x: x,
                    new_self.block_widths,
                    transposed=other_is_transposed,
                )
            else:
                new_other = other
        # Most of the time, we will be given an operation to do. We perform that with
        # manual_shuffle.
        else:
            new_other = other.manual_shuffle(
                axis,
                right_func,
                new_self.block_lengths if axis == 0 else new_self.block_widths,
                transposed=other_is_transposed,
            )
        return new_self, new_other

    @classmethod
    def map_across_full_axis(cls, axis, partitions, map_func):
        """Applies `map_func` to every partition.

        Note: This method should be used in the case that `map_func` relies on
            some global information about the axis.

        Args:
            axis: The axis to perform the map across (0 - index, 1 - columns).
            map_func: The function to apply.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        # Since we are already splitting the DataFrame back up after an
        # operation, we will just use this time to compute the number of
        # partitions as best we can right now.
        num_splits = cls._compute_num_partitions()
        preprocessed_map_func = cls.preprocess_func(map_func)
        partitions = cls.column_partitions(partitions) if not axis else cls.row_partitions(partitions)
        # For mapping across the entire axis, we don't maintain partitioning because we
        # may want to line to partitioning up with another BlockPartitions object. Since
        # we don't need to maintain the partitioning, this gives us the opportunity to
        # load-balance the data as well.
        result_blocks = np.array(
            [
                part.apply(preprocessed_map_func, num_splits=num_splits)
                for part in partitions
            ]
        )
        # If we are mapping over columns, they are returned to use the same as
        # rows, so we need to transpose the returned 2D numpy array to return
        # the structure to the correct order.
        return result_blocks.T if not axis else result_blocks

    @classmethod
    def take(cls, axis, partitions, block_lengths, n):
        """Take the first (or last) n rows or columns from the blocks

        Note: Axis = 0 will be equivalent to `head` or `tail`
              Axis = 1 will be equivalent to `front` or `back`

        Args:
            axis: The axis to extract (0 for extracting rows, 1 for extracting columns)
            n: The number of rows or columns to extract, negative denotes to extract
                from the bottom of the object

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        # These are the partitions that we will extract over
        if not axis:
            partitions = partitions
            bin_lengths = block_lengths
        else:
            partitions = partitions.T
            bin_lengths = block_lengths
        if n < 0:
            length_bins = np.cumsum(bin_lengths[::-1])
            n *= -1
            idx = int(np.digitize(n, length_bins))
            if idx > 0:
                remaining = int(n - length_bins[idx - 1])
            else:
                remaining = n
            # In this case, we require no remote compute. This is much faster.
            if remaining == 0:
                result = partitions[-idx:]
            else:
                # Reverse for ease of iteration and then re-reverse at the end
                partitions = partitions[::-1]
                # We build this iloc to avoid creating a bunch of helper methods.
                # This code creates slice objects to be passed to `iloc` to grab
                # the last n rows or columns depending on axis.
                slice_obj = (
                    slice(-remaining, None)
                    if axis == 0
                    else (slice(None), slice(-remaining, None))
                )
                func = cls.preprocess_func(lambda df: df.iloc[slice_obj])
                # We use idx + 1 here because the loop is not inclusive, and we
                # need to iterate through idx.
                result = np.array(
                    [
                        partitions[i]
                        if i != idx
                        else [obj.apply(func) for obj in partitions[i]]
                        for i in range(idx + 1)
                    ]
                )[::-1]
        else:
            length_bins = np.cumsum(bin_lengths)
            idx = int(np.digitize(n, length_bins))
            if idx > 0:
                remaining = int(n - length_bins[idx - 1])
            else:
                remaining = n
            # In this case, we require no remote compute. This is much faster.
            if remaining == 0:
                result = partitions[:idx]
            else:
                # We build this iloc to avoid creating a bunch of helper methods.
                # This code creates slice objects to be passed to `iloc` to grab
                # the first n rows or columns depending on axis.
                slice_obj = (
                    slice(remaining) if axis == 0 else (slice(None), slice(remaining))
                )
                func = cls.preprocess_func(lambda df: df.iloc[slice_obj])
                # See note above about idx + 1
                result = np.array(
                    [
                        partitions[i]
                        if i != idx
                        else [obj.apply(func) for obj in partitions[i]]
                        for i in range(idx + 1)
                    ]
                )
        return result.T if axis else result

    def concat(self, axis, other_blocks):
        """Concatenate the blocks with another set of blocks.

        Note: Assumes that the blocks are already the same shape on the
            dimension being concatenated. A ValueError will be thrown if this
            condition is not met.

        Args:
            axis: The axis to concatenate to.
            other_blocks: the other blocks to be concatenated. This is a
                BaseFrameManager object.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if type(other_blocks) is list:
            other_blocks = [blocks.partitions for blocks in other_blocks]
            return self.__constructor__(
                np.concatenate([self.partitions] + other_blocks, axis=axis)
            )
        else:
            return self.__constructor__(
                np.append(self.partitions, other_blocks.partitions, axis=axis)
            )

    def copy(self):
        """Create a copy of this object.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        return self.__constructor__(self.partitions.copy())

    def transpose(self, *args, **kwargs):
        """Transpose the blocks stored in this object.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        return self.__constructor__(self.partitions.T)

    @classmethod
    def to_pandas(cls, partitions):
        """Convert this object into a Pandas DataFrame from the partitions.

        Returns:
            A Pandas DataFrame
        """
        retrieved_objects = [
            [obj.to_pandas() for obj in part]
            for part in partitions
        ]
        if all(
            isinstance(part, pandas.Series)
            for row in retrieved_objects
            for part in row
        ):
            axis = 0
        elif all(
            isinstance(part, pandas.DataFrame)
            for row in retrieved_objects
            for part in row
        ):
            axis = 1
        else:
            ErrorMessage.catch_bugs_and_request_email(True)
        df_rows = [
            pandas.concat([part for part in row], axis=axis)
            for row in retrieved_objects
            if not all(part.empty for part in row)
        ]
        if len(df_rows) == 0:
            return pandas.DataFrame()
        else:
            return pandas.concat(df_rows)

    def to_numpy(self, is_transposed=False):
        """Convert this object into a NumPy Array from the partitions.

        Returns:
            A NumPy Array
        """
        arr = np.block([[block.to_numpy() for block in row] for row in self.partitions])
        if is_transposed:
            return arr.T
        return arr

    @classmethod
    def from_pandas(cls, df):
        num_splits = cls._compute_num_partitions()
        put_func = cls._partition_class.put
        row_chunksize, col_chunksize = compute_chunksize(df, num_splits)

        # Each chunk must have a RangeIndex that spans its length and width
        # according to our invariant.
        def chunk_builder(i, j):
            chunk = df.iloc[i : i + row_chunksize, j : j + col_chunksize].copy()
            chunk.index = pandas.RangeIndex(len(chunk.index))
            chunk.columns = pandas.RangeIndex(len(chunk.columns))
            return put_func(chunk)

        parts = [
            [chunk_builder(i, j) for j in range(0, len(df.columns), col_chunksize)]
            for i in range(0, len(df), row_chunksize)
        ]
        return cls(np.array(parts))

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """This gets the internal indices stored in the partitions.

        Note: These are the global indices of the object. This is mostly useful
            when you have deleted rows/columns internally, but do not know
            which ones were deleted.

        Args:
            axis: This axis to extract the labels. (0 - index, 1 - columns).
            index_func: The function to be used to extract the function.
            old_blocks: An optional previous object that this object was
                created from. This is used to compute the correct offsets.

        Returns:
            A Pandas Index object.
        """
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        if axis == 0:
            # We grab the first column of blocks and extract the indices
            # Note: We use _partitions_cache in the context of this function to make
            # sure that none of the partitions are modified or filtered out before we
            # get the index information.
            # DO NOT CHANGE TO self.partitions under any circumstance.
            new_idx = (
                [idx.apply(func).get() for idx in partitions.T[0]]
                if len(partitions.T)
                else []
            )
        else:
            new_idx = (
                [idx.apply(func).get() for idx in partitions[0]]
                if len(partitions)
                else []
            )
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

    @classmethod
    def _compute_num_partitions(cls):
        """Currently, this method returns the default. In the future it will
            estimate the optimal number of partitions.

        :return:
        """
        from modin.pandas import DEFAULT_NPARTITIONS

        return DEFAULT_NPARTITIONS

    # Extracting rows/columns
    def _get_blocks_containing_index(self, axis, index):
        """Convert a global index to a block index and local index.

        Note: This method is primarily used to convert a global index into a
            partition index (along the axis provided) and local index (useful
            for `iloc` or similar operations.

        Args:
            axis: The axis along which to get the indices
                (0 - columns, 1 - rows)
            index: The global index to convert.

        Returns:
            A tuple containing (block index and internal index).
        """
        if not axis:
            ErrorMessage.catch_bugs_and_request_email(index > sum(self.block_widths))
            cumulative_column_widths = np.array(self.block_widths).cumsum()
            block_idx = int(np.digitize(index, cumulative_column_widths))
            if block_idx == len(cumulative_column_widths):
                block_idx -= 1
            # Compute the internal index based on the previous lengths. This
            # is a global index, so we must subtract the lengths first.
            internal_idx = (
                index
                if not block_idx
                else index - cumulative_column_widths[block_idx - 1]
            )
        else:
            ErrorMessage.catch_bugs_and_request_email(index > sum(self.block_lengths))
            cumulative_row_lengths = np.array(self.block_lengths).cumsum()
            block_idx = int(np.digitize(index, cumulative_row_lengths))
            # See note above about internal index
            internal_idx = (
                index
                if not block_idx
                else index - cumulative_row_lengths[block_idx - 1]
            )
        return block_idx, internal_idx

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """Applies a function to a list of remote partitions.

        Note: The main use for this is to preprocess the func.

        Args:
            func: The func to apply
            partitions: The list of partitions

        Returns:
            A list of BaseFramePartition objects.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]

    def apply_func_to_select_indices(self, axis, func, indices, keep_remaining=False):
        """Applies a function to select indices.

        Note: Your internal function must take a kwarg `internal_indices` for
            this to work correctly. This prevents information leakage of the
            internal index to the external representation.

        Args:
            axis: The axis to apply the func over.
            func: The function to apply to these indices.
            indices: The indices to apply the function to.
            keep_remaining: Whether or not to keep the other partitions.
                Some operations may want to drop the remaining partitions and
                keep only the results.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if self.partitions.size == 0:
            return np.array([[]])
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        if not axis:
            partitions_for_apply = self.partitions.T
        else:
            partitions_for_apply = self.partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:

            def local_to_global_idx(partition_id, local_idx):
                if partition_id == 0:
                    return local_idx
                if axis == 0:
                    cumulative_axis = np.cumsum(self.block_widths)
                else:
                    cumulative_axis = np.cumsum(self.block_lengths)
                return cumulative_axis[partition_id - 1] + local_idx

            if not keep_remaining:
                result = np.array(
                    [
                        self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[o_idx],
                            func_dict={
                                i_idx: dict_func[local_to_global_idx(o_idx, i_idx)]
                                for i_idx in list_to_apply
                                if i_idx >= 0
                            },
                        )
                        for o_idx, list_to_apply in indices
                    ]
                )
            else:
                result = np.array(
                    [
                        partitions_for_apply[i]
                        if i not in indices
                        else self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_func[local_to_global_idx(i, idx)]
                                for idx in indices[i]
                                if idx >= 0
                            },
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        else:
            if not keep_remaining:
                # We are passing internal indices in here. In order for func to
                # actually be able to use this information, it must be able to take in
                # the internal indices. This might mean an iloc in the case of Pandas
                # or some other way to index into the internal representation.
                result = np.array(
                    [
                        self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[idx],
                            internal_indices=list_to_apply,
                        )
                        for idx, list_to_apply in indices
                    ]
                )
            else:
                # The difference here is that we modify a subset and return the
                # remaining (non-updated) blocks in their original position.
                result = np.array(
                    [
                        partitions_for_apply[i]
                        if i not in indices
                        else self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            internal_indices=indices[i],
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        return (
            self.__constructor__(result.T) if not axis else self.__constructor__(result)
        )

    @classmethod
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """Applies a function to a select subset of full columns/rows.

        Note: This should be used when you need to apply a function that relies
            on some global information for the entire column/row, but only need
            to apply a function to a subset.

        Important: For your func to operate directly on the indices provided,
            it must use `internal_indices` as a keyword argument.

        Args:
            axis: The axis to apply the function over (0 - rows, 1 - columns)
            func: The function to apply.
            indices: The global indices to apply the func to.
            keep_remaining: Whether or not to keep the other partitions.
                Some operations may want to drop the remaining partitions and
                keep only the results.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if partitions.size == 0:
            return np.array([[]])
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        preprocessed_func = cls.preprocess_func(func)
        # Since we might be keeping the remaining blocks that are not modified,
        # we have to also keep the block_partitions object in the correct
        # direction (transpose for columns).
        if not axis:
            partitions_for_apply = cls.column_partitions(partitions)
            partitions_for_remaining = partitions.T
        else:
            partitions_for_apply = cls.row_partitions(partitions)
            partitions_for_remaining = partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        partitions_for_apply[i].apply(
                            preprocessed_func,
                            func_dict={
                                idx: dict_func[idx] for idx in indices[i]
                            },
                        )
                        for i in indices
                    ]
                )
            else:
                result = np.array(
                    [
                        partitions_for_remaining[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            preprocessed_func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_func[idx] for idx in indices[i]
                            },
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        else:
            if not keep_remaining:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        partitions_for_apply[i].apply(
                            preprocessed_func, internal_indices=indices[i]
                        )
                        for i in indices
                    ]
                )
            else:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        partitions_for_remaining[i]
                        if i not in indices
                        else partitions_for_apply[i].apply(
                            preprocessed_func, internal_indices=indices[i]
                        )
                        for i in range(len(partitions_for_remaining))
                    ]
                )
        return result.T if not axis else result

    def mask(self, row_indices=None, col_indices=None):
        ErrorMessage.catch_bugs_and_request_email(
            row_indices is None and col_indices is None
        )
        if row_indices is not None:
            row_partitions_list = self._get_dict_of_block_index(
                1, row_indices, ordered=True
            )
        else:
            row_partitions_list = [
                (i, range(self.block_lengths[i]))
                for i in range(len(self.block_lengths))
            ]

        if col_indices is not None:
            col_partitions_list = self._get_dict_of_block_index(
                0, col_indices, ordered=True
            )
        else:
            col_partitions_list = [
                (i, range(self.block_widths[i])) for i in range(len(self.block_widths))
            ]
        return self.__constructor__(
            np.array(
                [
                    [
                        self.partitions[row_idx][col_idx].mask(
                            row_internal_indices, col_internal_indices
                        )
                        for col_idx, col_internal_indices in col_partitions_list
                        if len(col_internal_indices) > 0
                    ]
                    for row_idx, row_internal_indices in row_partitions_list
                    if len(row_internal_indices) > 0
                ]
            )
        )

    def apply_func_to_indices_both_axis(
        self,
        func,
        row_indices,
        col_indices,
        lazy=False,
        keep_remaining=True,
        mutate=False,
        item_to_distribute=None,
    ):
        """
        Apply a function to along both axis

        Important: For your func to operate directly on the indices provided,
            it must use `row_internal_indices, col_internal_indices` as keyword
            arguments.
        """
        if keep_remaining:
            row_partitions_list = self._get_dict_of_block_index(1, row_indices).items()
            col_partitions_list = self._get_dict_of_block_index(0, col_indices).items()
        else:
            row_partitions_list = self._get_dict_of_block_index(
                1, row_indices, ordered=True
            )
            col_partitions_list = self._get_dict_of_block_index(
                0, col_indices, ordered=True
            )
            result = np.empty(
                (len(row_partitions_list), len(col_partitions_list)), dtype=type(self)
            )

        if not mutate:
            partition_copy = self.partitions.copy()
        else:
            partition_copy = self.partitions

        row_position_counter = 0
        for row_idx, row_values in enumerate(row_partitions_list):
            row_blk_idx, row_internal_idx = row_values
            col_position_counter = 0
            for col_idx, col_values in enumerate(col_partitions_list):
                col_blk_idx, col_internal_idx = col_values
                remote_part = partition_copy[row_blk_idx, col_blk_idx]

                if item_to_distribute is not None:
                    item = item_to_distribute[
                        row_position_counter : row_position_counter
                        + len(row_internal_idx),
                        col_position_counter : col_position_counter
                        + len(col_internal_idx),
                    ]
                    item = {"item": item}
                else:
                    item = {}

                if lazy:
                    block_result = remote_part.add_to_apply_calls(
                        func,
                        row_internal_indices=row_internal_idx,
                        col_internal_indices=col_internal_idx,
                        **item
                    )
                else:
                    block_result = remote_part.apply(
                        func,
                        row_internal_indices=row_internal_idx,
                        col_internal_indices=col_internal_idx,
                        **item
                    )
                if keep_remaining:
                    partition_copy[row_blk_idx, col_blk_idx] = block_result
                else:
                    result[row_idx][col_idx] = block_result
                col_position_counter += len(col_internal_idx)

            row_position_counter += len(row_internal_idx)

        if keep_remaining:
            return self.__constructor__(partition_copy)
        else:
            return self.__constructor__(result)

    def inter_data_operation(self, axis, func, other):
        """Apply a function that requires two BaseFrameManager objects.

        Args:
            axis: The axis to apply the function over (0 - rows, 1 - columns)
            func: The function to apply
            other: The other BaseFrameManager object to apply func to.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if axis:
            partitions = self.row_partitions
            other_partitions = other.row_partitions
        else:
            partitions = self.column_partitions
            other_partitions = other.column_partitions
        func = self.preprocess_func(func)
        result = np.array(
            [
                partitions[i].apply(
                    func,
                    num_splits=self._compute_num_partitions(),
                    other_axis_partition=other_partitions[i],
                )
                for i in range(len(partitions))
            ]
        )
        return self.__constructor__(result) if axis else self.__constructor__(result.T)

    def manual_shuffle(self, axis, shuffle_func, lengths, transposed=False):
        """Shuffle the partitions based on the `shuffle_func`.

        Args:
            axis: The axis to shuffle across.
            shuffle_func: The function to apply before splitting the result.
            lengths: The length of each partition to split the result into.

        Returns:
             A new BaseFrameManager object, the type of object that called this.
        """
        if axis:
            partitions = self.row_partitions
        else:
            partitions = self.column_partitions
        func = self.preprocess_func(shuffle_func)
        result = np.array(
            [part.shuffle(func, lengths, _transposed=transposed) for part in partitions]
        )
        return self.__constructor__(result) if axis else self.__constructor__(result.T)

    def __getitem__(self, key):
        return self.__constructor__(self.partitions[key])

    def __len__(self):
        return sum(self.block_lengths)

    def enlarge_partitions(self, n_rows=None, n_cols=None):
        data = self.partitions
        if n_rows:
            n_cols_lst = self.block_widths
            nan_oids_lst = [
                self._partition_class(
                    _get_nan_block_id(self._partition_class, n_rows, n_cols_)
                )
                for n_cols_ in n_cols_lst
            ]
            new_chunk = self.__constructor__(np.array([nan_oids_lst]))
            data = self.concat(axis=0, other_blocks=new_chunk)

        if n_cols:
            n_rows_lst = self.block_lengths
            nan_oids_lst = [
                self._partition_class(
                    _get_nan_block_id(self._partition_class, n_rows_, n_cols)
                )
                for n_rows_ in n_rows_lst
            ]
            new_chunk = self.__constructor__(np.array([nan_oids_lst]).T)
            data = self.concat(axis=1, other_blocks=new_chunk)
        return data
