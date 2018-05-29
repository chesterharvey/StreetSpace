################################################################################
# Module: utils.py
# Description: Miscellaneous tools
# License: MIT, see full license in LICENSE.txt
################################################################################

import numpy as np
import pandas as pd
import subprocess
import collections

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