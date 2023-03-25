from pathlib import Path
import re
import pandas as pd
import numpy as np
import logging
from ini_estim import DATA_DIR as _DATA_DIR


_LOG = logging.getLogger(__name__)
ROOT_DIR = Path(_DATA_DIR) / "deka"
_COLLECTIONS = {p.name: p for p in ROOT_DIR.iterdir() if
                not p.name.startswith(".") and p.is_dir()}

_REGOBJ = r"(?P<object>\w+(?=(_grip\d)|((?<!_grip\d)(_\d{4}))))"
_REGGRIP = r"\_(?P<grip>grip\d)?"
_REGDATE = r"_?(?P<date>[\d\-]+)\."
_REGEX_PATTERN = _REGOBJ + _REGGRIP + _REGDATE


def get_file_info():
    """ Returns a Pandas DataFrame with all the data file info """
    all_files = []
    for c, path in _COLLECTIONS.items():
        datafiles = path.glob("*.csv")

        for file in datafiles:
            m = re.match(_REGEX_PATTERN, file.name)

            d = m.groupdict()
            if d is None:
                _LOG.warning("Unable to parse: " + d.name)
                continue

            if d.get('grip') is None:
                d['grip'] = 'free'
            else:
                d['grip'] = d['grip'].strip('grip')

            datestr = d.get('date')
            if datestr is not None:
                date_tok = [int(tok) for tok in datestr.split('-')]
                try:
                    date = pd.Timestamp(
                        year=date_tok[0],
                        month=date_tok[1],
                        day=date_tok[2],
                        hour=date_tok[3],
                        minute=date_tok[4],
                        second=date_tok[5]
                    )
                except Exception as e:
                    print("EXCEPTION! {}".format(d))
                    raise
            else:
                date = pd.Timestamp()

            d['file'] = file.relative_to(ROOT_DIR).as_posix()
            d['collection'] = c
            d['date'] = date
            all_files.append(d)
    return pd.DataFrame.from_records(all_files)


class DEKAData:
    """ Convenience class for reading DEKA data

    Internally holds a Pandas dataframe generated from "get_file_info".
    Loads CSV files by indexing into the dataframe.
    """
    def __init__(self):
        self._df = get_file_info()
        self._objects = self._df['object'].unique()
        self._collections = self._df['collection'].unique()
        self._grips = self._df['grip'].unique()

    @property
    def objects(self):
        return self._objects

    @property
    def collections(self):
        return self._collections

    @property
    def grips(self):
        return self._grips

    def get_info(self, object_name=None, grip=None, collection=None):
        """ Get a sub-dataframe that matches the specified parameters

        Any parameter can be left as None (default), which will load all
        rows of that parameter.

        Parameters
        ----------
        object_name : str
            The object name. Found from self.objects
        grip : str
            The grip. Found from self.grips
        collection : str
            The data collection. Found from self.collections

        Returns
        -------
        pd.DataFrame
        """
        idx = np.ones(self._df.shape[0], dtype=np.bool)
        if object_name is not None:
            idx &= self._df['object'] == object_name
        if grip is not None:
            idx &= self._df['grip'] == str(grip)
        if collection is not None:
            idx &= self._df['collection'] == collection
        subdf = self._df[idx]
        return subdf

    def read_csv(self, index):
        """ Load a CSV file into a DataFrame

        Parameters
        ----------
        index
            The index of the corresponding row in the internal dataframe
            An example of loading the first csv file from a specific object:
            >>> d = DEKAData()
            >>> idx = d.get_info('plush_animal_bear').index[0]
            >>> df = d.read_csv(idx)

        Returns
        -------
        pd.DataFrame
        """
        if np.isscalar(index):
            return self._read_csv_single(index)
        else:
            dfs = [self._read_csv_single(i) for i in index]
            dfs = [df for df in dfs if df is not None]

            return pd.concat(dfs, keys=index)


    def _read_csv_single(self, index):
        try:
            file = ROOT_DIR / self._df.iloc[index]['file']
        except IndexError:
            return None
        else:
            return pd.read_csv(file, sep=' ', header=1)

