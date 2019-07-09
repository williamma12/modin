from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas


def get_default_chunksize(length, num_splits):
    """Creates the most equal chunksize possible based on length and number of splits.

    Args:
        length: The integer length to split (number of rows/columns).
        num_splits: The integer number of splits.

    Returns:
        An integer chunksize.
    """
    return (
        length // num_splits if length % num_splits == 0 else length // num_splits + 1
    )


def compute_chunksize(df, num_splits, default_block_size=32, axis=None):
    """Computes the number of rows and/or columns to include in each partition.

    Args:
        df: The DataFrame to split.
        num_splits: The maximum number of splits to separate the DataFrame into.
        default_block_size: Minimum number of rows/columns (default set to 32x32).
        axis: The axis to split. (0: Index, 1: Columns, None: Both)

    Returns:
         If axis is 1 or 0, returns an integer number of rows/columns to split the
         DataFrame. If axis is None, return a tuple containing both.
    """
    if axis == 0 or axis is None:
        row_chunksize = get_default_chunksize(len(df.index), num_splits)
        # Take the min of the default and the memory-usage chunksize first to avoid a
        # large amount of small partitions.
        row_chunksize = max(1, row_chunksize, default_block_size)
        if axis == 0:
            return row_chunksize
    # We always execute this because we can only get here if axis is 1 or None.
    col_chunksize = get_default_chunksize(len(df.columns), num_splits)
    # Take the min of the default and the memory-usage chunksize first to avoid a
    # large amount of small partitions.
    col_chunksize = max(1, col_chunksize, default_block_size)
    if axis == 1:
        return col_chunksize

    return row_chunksize, col_chunksize


def _get_nan_block_id(partition_class, n_row=1, n_col=1, transpose=False):
    """A memory efficient way to get a block of NaNs.

    Args:
        partition_class (BaseFramePartition): The class to use to put the object
            in the remote format.
        n_row(int): The number of rows.
        n_col(int): The number of columns.
        transpose(bool): If true, swap rows and columns.
    Returns:
        ObjectID of the NaN block.
    """
    global _NAN_BLOCKS
    if transpose:
        n_row, n_col = n_col, n_row
    shape = (n_row, n_col)
    if shape not in _NAN_BLOCKS:
        arr = np.tile(np.array(np.NaN), shape)
        # TODO Not use pandas.DataFrame here, but something more general.
        _NAN_BLOCKS[shape] = partition_class.put(pandas.DataFrame(data=arr))
    return _NAN_BLOCKS[shape]


def split_result_of_axis_func_pandas(axis, num_splits, result, length_list=None):
    """Split the Pandas result evenly based on the provided number of splits.

    Args:
        axis: The axis to split across.
        num_splits: The number of even splits to create.
        result: The result of the computation. This should be a Pandas
            DataFrame.
        length_list: The list of lengths to split this DataFrame into. This is used to
            return the DataFrame to its original partitioning schema.

    Returns:
        A list of Pandas DataFrames.
    """
    if num_splits == 1:
        return result
    if length_list is not None:
        length_list.insert(0, 0)
        sums = np.cumsum(length_list)
        if axis == 0:
            return [result.iloc[sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
        else:
            return [result.iloc[:, sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
    # We do this to restore block partitioning
    chunksize = compute_chunksize(result, num_splits, axis=axis)
    if axis == 0:
        return [
            result.iloc[chunksize * i : chunksize * (i + 1)] for i in range(num_splits)
        ]
    else:
        return [
            result.iloc[:, chunksize * i : chunksize * (i + 1)]
            for i in range(num_splits)
        ]


def length_fn_pandas(df):
    assert isinstance(df, pandas.DataFrame)
    return len(df)


def width_fn_pandas(df):
    assert isinstance(df, pandas.DataFrame)
    return len(df.columns)


def set_indices_for_pandas_concat(df, transposed=False):
    df.index = pandas.RangeIndex(len(df))
    df.columns = pandas.RangeIndex(len(df.columns))
    return df.T if transposed else df


def compute_partition_shuffle(old_lengths, new_lengths, old_index=None, new_index=None):
    """Calculates split old_length partitions into new_lengths partitions.

    Args:
        old_lengths: Lengths of the old partitions.
        new_lengths: Lengths of the new partitions.
        old_index: Current ordering of the labels.
        new_index: New ordering of the labels of the data.

    Returns:
        List containing a list for each of the new partitions. Each of the inner
        lists contain tuples contains the index of the original partition and a
        list of the indices of that partition that are in the new partition.
    """
    # Create a dataframe of index values to facilitate the shuffle calculations.
    # We use -1 to represent NaN indices.
    # The resulting dataframe will look like the following
    #   old_block_index     old_internal_index  new_block_index
    # 0     0                   0                   0
    # 1     0                   1                   0
    # 2     1                   0                   0
    # 3     1                   1                   1
    # 4     -1                  -1                  1
    data = {"old_block_index": [], "old_internal_index": []}
    new_block_index_col = []
    for i, length in enumerate(old_lengths):
        data["old_internal_index"].extend([j for j in range(length)])
        data["old_block_index"].extend([i for _ in range(length)])
    for i, length in enumerate(new_lengths):
        new_block_index_col.extend([i for _ in range(length)])
    if new_index is not None and len(new_index) > len(old_index):
        diff = len(new_index) - len(old_index)
        print(diff)
        data["old_block_index"].append([-1 for _ in range(diff)])
        data["old_internal_index"].append([-1 for _ in range(diff)])
        old_index = old_index.append(pandas.Index([-(i+1) for i in range(diff)]))

    # Create index dataframe and compute reindex
    index_df = pandas.DataFrame(data, index=old_index)
    if new_index is not None:
        index_df = index_df.reindex(new_index).fillna(-1).astype(int)
    index_df["new_block_index"] = new_block_index_col

    # Using the dataframe, we iterate through the dataframe to get a list of
    # indices old block index and a list of old block indices in the new 
    # partition.
    result = []
    block_result = []
    internal_block_result = []
    prev_old_block = 0
    prev_new_block = 0
    for _, row in index_df.iterrows():
        old_block_idx, old_internal_idx, new_block_idx = row
        if new_block_idx != prev_new_block:
            block_result.append(internal_block_result)
            result.append(block_result)
            internal_block_result = []
            block_result = []
            prev_new_block = new_block_idx
            
        if old_block_idx == prev_old_block and len(internal_block_result) > 0:
            internal_block_result[1].append(old_internal_idx)
        else:
            if len(internal_block_result) > 0:
                block_result.append(internal_block_result)
            prev_old_block = old_block_idx
            internal_block_result = [old_block_idx, [old_internal_idx]]
    if len(internal_block_result) > 0:
        block_result.append(internal_block_result)
    if len(block_result) > 0:
        result.append(block_result)
    return result
