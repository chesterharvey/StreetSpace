################################################################################
# Module: utils.py
# Description: Miscellaneous tools
# License: MIT, see full license in LICENSE.txt
################################################################################

import numpy as np
import subprocess

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


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
