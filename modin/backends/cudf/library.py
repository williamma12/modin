import pandas

def _isna_builder(df, *args, **kwargs):
    return pandas.DataFrame.isna(df, *args, **kwargs)

isna = _isna_builder

_conversion = lambda x: x
