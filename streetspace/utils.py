################################################################################
# Module: utils.py
# Description: Miscellaneous tools
# License: MIT, see full license in LICENSE.txt
################################################################################

import numpy as np
import pandas as pd
import subprocess
import collections
import sys
from matplotlib.colors import LinearSegmentedColormap

def listify(x):
    """Puts non-list objects into a list. 

    Parameters
    ----------
    x: Object to ensure is list
    
    Returns
    ----------
    :obj:`list`
        If ``x`` is already a list, this is returned. Otherwise, a list\
        containing ``x`` is returned.
    """
    if isinstance(x, list):
        return x
    else:
        return [x]


def unlistify(x):
    """Unpacks single-object lists into their internal object

    If list is longer than one, returns original list

    Parameters
    ----------
    x: :obj:`list`
        Input list
    
    Returns
    ----------
    :obj:`list` or other non-list object
        If ``len(x) == 1``, returns the single object in ``x``.
        Otherwise, returns ``x``
    """
    if isinstance(x, list):
        if len(x) == 1:
            return x[0]
    return x


def unpack_nested_lists(x):
    """Unpacks listed lists and non-lists into a single, unnested list
    
    Parameters
    ----------
    x :obj:`list`
        List containing lists and non-lists
    
    Returns
    ----------
    :obj:`list`
        Any lists within ``x`` will be unpacked as individual items
        alongside all the other items within ``x``

    """
    return [i if isinstance(j, list) else j for j in x for i in j]


def bash_command(command, verbose=False):
    """
    Executes a bash command passed as a string

    Parameters
    ----------
    command: str
        bash command
    
    verbose: bool
        True = print available output and error messages
        False (default)

    Returns
    ----------
    (outputs, errors): (str, str)
    """

    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True,
                         universal_newlines=True)
    output, error = p.communicate()
    if verbose:
        if bool(output) > 0:
            print(output)
        if bool(error) > 0:
            print('error: {}'.format(error))


def odrive_install(verbose=False):
    """
    Wrapper for bash command to install the odrive sync agent to
    ~/.odrive-agent

    Parameters
    ----------
    verbose: bool
        True = print available output and error messages
        False (default)
    """

    command = 'od="$HOME/.odrive-agent/bin" && curl -L "https://dl.odrive.com/odrive-py" --create-dirs -o "$od/odrive.py" && curl -L "https://dl.odrive.com/odriveagent-lnx-32" | tar -xvzf- -C "$od/" && curl -L "https://dl.odrive.com/odrivecli-lnx-32" | tar -xvzf- -C "$od/"'
    bash_command(command, verbose=verbose)
    # p = subprocess.Popen(command, stdout=subprocess.PIPE,
    #                      stderr=subprocess.PIPE, shell=True,
    #                      universal_newlines=True)
    # output, error = p.communicate()
    # if verbose:
    #     if bool(output) > 0:
    #         print(output)
    #     if bool(error) > 0:
    #         print('error: {}'.format(error))


def odrive_run(verbose=False):
    """
    Wrapper for bash command to run the odrive sync agent server in the background.

    Parameters
    ----------
    verbose: bool
        True = print available output and error messages
        False (default)

    Returns
    ----------
    (outputs, errors): (str, str)
    """

    command = 'nohup "$HOME/.odrive-agent/bin/odriveagent" > /dev/null 2>&1 &'
    bash_command(command, verbose=verbose)
    # p = subprocess.Popen(command, stdout=subprocess.PIPE,
    #                      stderr=subprocess.PIPE, shell=True,
    #                      universal_newlines=True)
    # output, error = p.communicate()
    # return output, error
    # if verbose:
    #     if bool(output) > 0:
    #         print(output)
    #     if bool(error) > 0:
    #         print('error: {}'.format(error))


def odrive_auth(key, verbose=False):
    """
    Wrapper for bash command to run the odrive sync agent server in the background.

    Parameters
    ----------
    key: str
        authentification key

    verbose: bool
        True = print available output and error messages
        False (default)

    Returns
    ----------
    (outputs, errors): (str, str)
    """

    command = 'python "$HOME/.odrive-agent/bin/odrive.py" authenticate {}'.format(key)
    bash_command(command, verbose=verbose)
    # p = subprocess.Popen(command, stdout=subprocess.PIPE,
    #                      stderr=subprocess.PIPE, shell=True,
    #                      universal_newlines=True)
    # output, error = p.communicate()
    # return output, error
    # if verbose:
    #     if bool(output) > 0:
    #         print(output)
    #     if bool(error) > 0:
    #         print('error: {}'.format(error))

def odrive_command(path, odrive_command='sync', verbose=False):
    """
    Wrapper for bash commands to operate the odrive sync agent's
    syncing commands.

    Parameters
    ----------
    path: str
        path for file or directory to sync

    command: str
        'sync' (default)
        'refresh'
        'unsync'

    Returns
    ----------
    (outputs, errors): (str, str)
    """
    odrive_py = '"$HOME/.odrive-agent/bin/odrive.py"'
    command = 'python {} {} {}'.format(odrive_py, odrive_command, path)
    bash_command(command, verbose=verbose)
    # p = subprocess.Popen(command, stdout=subprocess.PIPE,
    #                      stderr=subprocess.PIPE, shell=True,
    #                      universal_newlines=True)
    # output, error = p.communicate()
    # return output, error
    # if verbose:
    #     if bool(output) > 0:
    #         print(output)
    #     if bool(error) > 0:
    #         print('error: {}'.format(error))


def odrive_sync(path, verbose=False):
    """Executes odrive_command function with odrive_command = sync."""

    odrive_command(path, odrive_command='sync', verbose=verbose)


def odrive_refresh(path, verbose=False):
    """Executes odrive_command function with odrive_command = refresh."""

    odrive_command(path, odrive_command='refresh', verbose=verbose)


def odrive_unsync(path, verbose=False):
    """Executes odrive_command function with odrive_command = unsync."""

    odrive_command(path, odrive_command='unsync', verbose=verbose)


def odrive_status(verbose=False):
    """
    Wrapper for bash command to call the odrive sync agent's status command.

    Parameters
    ----------
    verbose: bool
        True = print available output and error messages
        False (default)

    Returns
    ----------
    (outputs, errors): (str, str)
    """

    command = 'python "$HOME/.odrive-agent/bin/odrive.py" status'
    bash_command(command, verbose=verbose)
    

def empty_array(rows, fields):
    """Initiate an empty array from a dictionary of field names and dtypes.

    Parameters
    ----------
    rows: :obj:`int`
        Number of rows
    fields: :obj:`dict`
        * Each field's name and dtype should be specified as a key-value pair.
        * Both name and dtype must be :obj:`str`

    Returns
    ----------
    :class:`numpy.ndarray`
        Empty array
    """
    return np.empty(rows, 
                    dtype={'names':tuple(fields.keys()),
                           'formats':tuple(fields.values())})


def df_first_column(df, column_name):
    """Move a dataframe column to the left-most position.
    
    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Dataframe to rearrange
        
    column_name: :obj:`str`
        Column to move.
    Returns
    ----------
    :class:`pandas.DataFrame`
        Rearranged dataframe
    """
    cols = list(df)
    cols.insert(0, cols.pop(cols.index(column_name)))
    return df.loc[:, cols]


def df_last_column(df, column_name):
    """Move a dataframe column to the right-most position.
    
    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Dataframe to rearrange
        
    column_name: :obj:`str`
        Column to move.
    Returns
    ----------
    :class:`pandas.DataFrame`
        Rearranged dataframe
    """
    cols = list(df)
    cols.insert(len(cols), cols.pop(cols.index(column_name)))
    return df.loc[:, cols]


def df_move_column(df, column_name, new_location):
    """Move a dataframe column to a new location based on integer index
    """
    df = df.copy()
    columns = df.columns.tolist()
    columns.insert(new_location, columns.pop(columns.index(column_name)))
    return df[columns]


def first(list_to_summarize):
    """Get the first item in a list.
    """
    return list_to_summarize[0]

def last(list_to_summarize):
    """Get the last item in a list.
    """
    return list_to_summarize[-1]

def middle(list_to_summarize):
    """Get the middle item in a list.
    """
    return list_to_summarize[len(test) // 2]

def concatenate(list_to_summarize, separator=', '):
    """Concatenate strings in a list.

    List items that are not strigns will be converted to strings.
    """
    for i, x in enumerate(list_to_summarize):
        if not isinstance(x, str):
            list_to_summarize[i] = str(x)
    return separator.join(list_to_summarize)


def applymap_numeric_columns(df, func):
    """Map a function elementwise to numeric columns.
    
    All other columns are returned unchanged.
    """
    columns = df._get_numeric_data().columns
    df[columns] = df[columns].applymap(func)
    return df


def applymap_dtype_columns(df, func, dtypes):
    """Map a function elementwise to columns with certain dtypes.
    
    All other columns are returned unchanged.
    """
    columns = pd.select_dtypes(include=dtypes).columns
    df[columns] = df[columns].applymap(func)
    return df


def applymap_specific_columns(df, columns, func):
    """Map a function elementwise to specific columns.
    
    All other columns are returned unchanged.
    """
    df[columns] = df[columns].applymap(func)
    return df


def map_new_column(df, column, func):
    """Apply a function to a column.
    
    All other columns are returned unchanged.
    """
    df[column] = df.apply(func, axis=1)
    return df


def insert_dummies(df, field, prefix=False, drop_field=True, zero_value=0):
    """Create dummy fields and insert them in place of the original field.

    """
    df = df.copy()
    dummies = pd.get_dummies(df[field])
    if zero_value != 0:
        dummies = dummies.replace(0, zero_value)
    if prefix:
        dummies = dummies.add_prefix(prefix)
    orig_idx = df.columns.get_loc(field)
    for i, column in enumerate(dummies.columns):
        if column not in df.columns:
            df.insert(orig_idx + i, column, dummies[column])
    if drop_field:
        if field in df.columns:
            df = df.drop([field], axis=1)
    return df


def make_google_maps_url(lat, lon):
    url = 'http://maps.google.com/maps?q=LAT,LON'
    url = url.replace('LAT', str(round(lat, 5)))
    url = url.replace('LON', str(round(lon, 5)))
    return url

def zoom_axis(ax, extent, axis_off=True):
    """Set extents of MatPlotLib axis

    ax : Matplotlib axis

    extent : (minx, miny, maxx, maxy) tuple

    """
    minx, miny, maxx, maxy = extent
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    if axis_off:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])


def select_columns(df, columns, prefix=None, suffix=None):
    """Select columns from a dataframe if they are in the dataframe
    
    """
    columns = [x for x in columns if x in df.columns]
    df = df[columns].copy()
    if prefix:
        columns = [prefix + x for x in df.columns]
        df.columns = columns
    if suffix:
        columns = [x + suffix for x in df.columns]
        df.columns = columns 
    return df


def nan_any(input, true_value=True, false_value=False):
    """Test whether element contains any True values, excluding NaN
    """
    if not isinstance(input, collections.Iterable):
        input = listify(input)
    if any(np.nan_to_num(input)):
        return true_value
    else:
        return false_value


def merge_intervals(intervals):
    """Merge overlapping intervals defined by a list of tuples
    
    From https://codereview.stackexchange.com/questions/69242/merging-overlapping-intervals
    """
    sorted_by_lower_bound = sorted(intervals, key=lambda x: x[0])
    merged = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                # replace by merged interval
                merged[-1] = (lower[0], upper_bound)  
            else:
                merged.append(higher)
    return merged


def merge_dictionaries(dicts):
    """Collapse values for similar keys in a list of dictionaries.

    Merged values are stored in lists
    """
    dd = collections.defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            dd[key].append(value)
    return dd


def flatten(l):
    """Flatten nested iterables within lists

    l : list containing individual values or lists

    """
    for x in l:
        if isinstance(x, collections.Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def remove_sequential_duplicates(l):
    """Drops duplicate values from a list while maintaining list order

    l : list

    """
    seen = set()
    return [x for x in l if not (x in seen or seen.add(x))]


def no_space_column_names(df):
    """Replace spaces in column names with underscores    
    """
    df = df.copy()
    df.columns = [x.replace(' ', '_') for x in df.columns]
    return df


def df_split_lists_into_rows(df, list_column, keep_columns=False, keep_indices=False):
    """Splits a column with lists into rows
    
    Modified from https://gist.github.com/jlln/338b4b0b55bd6984f883#gistcomment-2676729
    
    df : Input dataframe
    list_column : Column containing lists
    keep_columns : Column name or list of column names to keep alongside the split lists
    keep_indices : If True, will keep indexes referencing original df rows and list positions.
        If a list with two items, will use these as the column names. 
    """
    
    # create a new dataframe with each item in a seperate column, dropping rows with missing values
    col_df = pd.DataFrame(df[list_column].dropna().tolist(),index=df[list_column].dropna().index)

    # create a series with columns stacked as rows         
    stacked = col_df.stack()

    # make into a dataframe
    new_df = pd.DataFrame(stacked, columns=[list_column])
    
    # Reset index
    new_df = new_df.reset_index()
    
    # Merge on old columns
    if keep_columns:
        keep_columns = listify(keep_columns)
        new_df = new_df.merge(df[keep_columns], left_on='level_0', right_index=True)
    
    # Rename index fields, if desired
    if keep_indices:
        if keep_indices == True:
            keep_indices = ['df_index', 'list_index']
        new_df = new_df.rename(columns={'level_0':keep_indices[0],'level_1':keep_indices[1]})
    else:
        new_df = new_df.drop(columns=['level_0','level_1'])
    
    return new_df


def df_split_lists_into_columns(df, list_column, new_column_names, delete_list_column=True):
    """Splits same-length lists within a column into seperate columns
    """
    df = df.copy()
    df[new_column_names] = pd.DataFrame(df[list_column].tolist(), index=df.index)
    if delete_list_column:
        df = df.drop(columns=[list_column])
    return df


def collapse_hierarchical_column_names(df, delimiter='_'):
    '''Collapse multi-level column names in Pandas DataFrame into single-level column names
    
    Adapted from: https://stackoverflow.com/questions/14507794/pandas-how-to-flatten-a-hierarchical-index-in-columns

    Each level of the of the original names is separated by the delimiter
    '''
    df = df.copy()
    df.columns = [delimiter.join((lambda x: (str(y) for y in x))(col)).rstrip(delimiter).strip() for col in df.columns.values]
    return df


def tiny():
    """Returns a really tiny positive float
    """
    return np.finfo(float).tiny


def giant():
    """returns a really giant positive float
    """
    return sys.float_info.max


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values
    
    From https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
    '''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values
    
    From https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
    '''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.
        
        From https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp