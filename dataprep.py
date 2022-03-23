import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd


session_break_delta = '30min'
invalid_rank = 0
terminal_item = -1


def read_data(fields=None):
    names = 'userID,Timestamp,Location,AppID,Traffic'.lower().split(',')
    cols = None
    if fields is not None:
        cols = list(map(names.index, fields))
    data = (
        pd.read_csv(
            'App_usage_trace.txt', header=None, sep=' ',
            names=names, usecols=cols,
            dtype={'timestamp': str}
        )
        .assign(
            timestamp=lambda x: pd.to_datetime(x['timestamp'], format="%Y%m%d%H%M%S")
        )
        .sort_values(['userid', 'timestamp'])
    )
    return data


def drop_consequtive_repeats(df, window=None):
    '''Drops all consecutive repeated events except the first one.
    If window is defined, repeated events outside the time window
    will be preserved, not dropped.
    Assumes data is sorted by users and timestamps.'''
    reps = (df['appid'].diff() == 0) & (df['userid'].diff() == 0)
    if window is not None:
        window = pd.Timedelta(window)
        reps = reps & (df['timestamp'].diff() <= window)
    return df.drop(reps.index[reps])


def assign_session_id(df, session_break_delta):
    '''Assumes data is sorted by users and timestamps.'''
    return df.assign(
        sessid=lambda x: ( # split into sessions based on time interval
                (
                    (x['timestamp'].diff() > pd.Timedelta(session_break_delta)) 
                    & (x['userid'].diff() == 0)
                )
            ).groupby(x['userid']).cumsum()
    )


def select_previous_items(data, itemid, level):
    prev_items_data = (
        data[itemid]
        .shift(1, fill_value=terminal_item)
    )
    prev_items_data.loc[data[level].diff()!=0] = terminal_item # exclude inter-level connections
    return prev_items_data   

def generate_transition_matrices(data, shape=None, level=None, userid='userid', itemid='appid', exclude_item=None):
    '''
    `level` can be e.g. users or sessions. Defaults to users.
    '''    
    level = level or userid
    prev_items = select_previous_items(data, itemid, level)
    
    transition_df = pd.DataFrame(
        {'_to': data[itemid], '_from': prev_items},
        index=data.index, copy=False
    )
    items_mask = slice(None)
    if exclude_item is not None:
        items_mask = prev_items != exclude_item # e.g., skip breaks 
    if shape is None:
        shape = tuple(transition_df.loc[items_mask].max()[['_to', '_from']]+1)
    
    transition_data = {}
    grouper = data[level] if level == userid else [data[userid], data[level]]
    for group_id, item_data in transition_df.loc[items_mask].groupby(grouper):
        transitions = (
            item_data
            .groupby('_to')['_from'] # form single-item groups (i.e. only consider most recent items)
            .value_counts(normalize=True) # group frequencies
            .pow(0.5)
        )
        rows = transitions.index.get_level_values(0)
        cols = transitions.index.get_level_values(1)
        vals = transitions.values
        transition_data[group_id] = csr_matrix((vals, (rows, cols)), shape=shape)
    return transition_data


def index_session_breaks(df, time_delta=None):
    '''Assumes data is sorted by users and timestamps.'''
    time_delta = time_delta or session_break_delta
    return (
        (df['timestamp'].diff() > pd.Timedelta(time_delta)) &
        (df['userid'].diff() == 0)
    ).loc[lambda x: x].index


def add_session_breaks(df, session_break_delta):
    '''Assumes data is sorted by users and timestamps.'''
    df = df.reset_index(drop=True)
    sess_break_idx = index_session_breaks(df, time_delta=session_break_delta)
    breaks = (
        df.loc[sess_break_idx-1] # shift 1 step back, reuse context of previous app
        .assign(
            timestamp=lambda x: x.timestamp + pd.Timedelta(session_break_delta),
            appid = terminal_item, # "terminal" item id
            traffic = 0
        )
    )

    df_with_breaks = (
        pd.concat([df, breaks], ignore_index=True)
        .sort_values(['userid', 'timestamp'])
    )
    return df_with_breaks


def assign_positions(df, group_key):
    '''Assumes data is sorted properly.'''
    return df.assign(position=lambda x: x.groupby(group_key).cumcount(ascending=False))


def frequency_weight(x, coef=1, denom=1):
    return 1 + coef * np.log(1+x/denom)


def matrix_from_observations(data, row_id, col_id, value_id=None, *, idx_map=None, shape=None, dtype=None):
    if idx_map is not None:
        row_idx_map = pd.Index(idx_map[row_id], name=row_id)
        col_idx_map = pd.Index(idx_map[col_id], name=col_id)
        row_idx = row_idx_map.get_indexer_for(data[row_id])
        col_idx = col_idx_map.get_indexer_for(data[col_id])
    else:
        row_data = data[row_id].astype("category")
        col_data = data[col_id].astype("category")

        row_idx = row_data.cat.codes
        col_idx = col_data.cat.codes
        row_idx_map = row_data.cat.categories.rename(row_id)
        col_idx_map = col_data.cat.categories.rename(col_id)

    if value_id is None:
        values = np.ones_like(col_idx, dtype=dtype)
    else:
        values = data[value_id].values
    
    matrix = csr_matrix((values, (row_idx, col_idx)), shape=shape, dtype=dtype)
    return matrix, row_idx_map, col_idx_map


def list_session_lists(data, userid='userid', itemid='appid', sessid='sessid', min_length=0):
    """
    Assumes `data` is sorted chronologically.
    """
    return (
        data
        .groupby([userid, sessid])[itemid]
        .apply(list) # represent sessions as lists of items
        .loc[lambda x: x.apply(len)>=min_length] # filter sessions with <`min_length` items
        .groupby(level=userid)
        .apply(list) # represent users as lists of sessions
    )
