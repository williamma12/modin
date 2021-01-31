# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
import re
from typing import List

from modin.config import Backend

S3_ADDRESS_REGEX = re.compile("[sS]3://(.*?)/(.*)")
NOT_IMPLEMENTED_MESSAGE = "Implement in children classes!"


class FileDispatcher:
    frame_cls = None
    frame_partition_cls = None
    query_compiler_cls = None

    @classmethod
    def read(cls, *args, **kwargs):
        query_compiler = cls._read(*args, **kwargs)
        # TODO (devin-petersohn): Make this section more general for non-pandas kernel
        # implementations.
        if Backend.get() != "Pandas":
            raise NotImplementedError("FIXME")
        import pandas

        if hasattr(query_compiler, "dtypes") and any(
            isinstance(t, pandas.CategoricalDtype) for t in query_compiler.dtypes
        ):
            dtypes = query_compiler.dtypes
            return query_compiler.astype(
                {
                    t: dtypes[t]
                    for t in dtypes.index
                    if isinstance(dtypes[t], pandas.CategoricalDtype)
                }
            )
        return query_compiler

    @classmethod
    def _read(cls, *args, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def get_path(cls, file_path: str) -> str:
        """
        Returns the path of the file(s).

        Parameters
        ----------
        file_path: str
            String representing a path.

        Returns
        -------
        str
            String of strings of absolute file paths.
        """
        if S3_ADDRESS_REGEX.search(file_path):
            return cls._s3_path(file_path, False)[0]
        else:
            return os.path.abspath(file_path)

    @classmethod
    def file_open(cls, file_path, mode="rb", compression="infer"):
        if isinstance(file_path, str):
            match = S3_ADDRESS_REGEX.search(file_path)
            if match is not None:
                if file_path[0] == "S":
                    file_path = "{}{}".format("s", file_path[1:])
                import s3fs as S3FS
                from botocore.exceptions import NoCredentialsError

                s3fs = S3FS.S3FileSystem(anon=False)
                try:
                    return s3fs.open(file_path)
                except NoCredentialsError:
                    s3fs = S3FS.S3FileSystem(anon=True)
                    return s3fs.open(file_path)
            elif compression == "gzip":
                import gzip

                return gzip.open(file_path, mode=mode)
            elif compression == "bz2":
                import bz2

                return bz2.BZ2File(file_path, mode=mode)
            elif compression == "xz":
                import lzma

                return lzma.LZMAFile(file_path, mode=mode)
            elif compression == "zip":
                import zipfile

                zf = zipfile.ZipFile(file_path, mode=mode.replace("b", ""))
                if zf.mode == "w":
                    return zf
                elif zf.mode == "r":
                    zip_names = zf.namelist()
                    if len(zip_names) == 1:
                        f = zf.open(zip_names.pop())
                        return f
                    elif len(zip_names) == 0:
                        raise ValueError(
                            "Zero files found in ZIP file {}".format(file_path)
                        )
                    else:
                        raise ValueError(
                            "Multiple files found in ZIP file."
                            " Only one file per ZIP: {}".format(zip_names)
                        )

        return open(file_path, mode=mode)

    @classmethod
    def file_size(cls, f):
        cur_pos = f.tell()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(cur_pos, os.SEEK_SET)
        return size

    @staticmethod
    def _s3_path(s3_path: str, glob: bool) -> List[str]:
        """
        Get s3 file paths.

        Parameters
        ----------
        s3_path: str
            String of s3 path.
        glob: bool
            True if the path is a glob path.

        Returns
        -------
        list[str]
            List of s3 files based on the path.
        """
        # S3FS does not allow captial S in s3 addresses.
        if s3_path[0] == "S":
            s3_path = "{}{}".format("s", s3_path[1:])

        import s3fs as S3FS
        from botocore.exceptions import NoCredentialsError

        def get_file_path(fs_handle) -> List[str]:
            if not glob:
                if fs_handle.exists(s3_path):
                    return [s3_path]
                else:
                    return []
            else:
                return fs_handle.glob(s3_path)

        s3fs = S3FS.S3FileSystem(anon=False)
        try:
            return get_file_path(s3fs)
        except NoCredentialsError:
            pass
        s3fs = S3FS.S3FileSystem(anon=True)
        return get_file_path(s3fs)

    @classmethod
    def file_exists(cls, file_path: str) -> bool:
        """
        Checks if the file_path leads to an existing file.

        Parameters
        ----------
        file_path: str
            String representing a path.

        Returns
        -------
        bool
            True if the file path is valid.
        """
        if isinstance(file_path, str):
            if S3_ADDRESS_REGEX.search(file_path):
                return len(cls._s3_path(file_path, False)) > 0
        return os.path.exists(file_path)

    @classmethod
    def deploy(cls, func, args, num_returns):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def parse(self, func, args, num_returns):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def materialize(cls, obj_id):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
