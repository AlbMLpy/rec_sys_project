import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import pandas as pd

import dataprep
from dataprep import drop_consequtive_repeats, assign_session_id, generate_transition_matrices, list_session_lists

from collections import Counter
from collections import defaultdict

class DataProcessor:
    """
        To fill in!
    """

    def __init__(self, file_name, column_names, userid='userid', itemid='appid', timeid='timestamp', session_break_delta='30min'):
        self.file = file_name
        self.names = column_names
        self.session_break_delta = session_break_delta
        self.userid = userid
        self.itemid = itemid
        self.timeid = timeid
        self.indexers = None
        self.n_users = None
        self.n_items = None
        self.data = None
        self.train, self.valid, self.test = None, None, None
        self.seen_interactions = None

    
    def read_data(self, usecols=None):
        """
            To fill in!
        """

        names = self.names.lower().split(',')
        data = (
            pd.read_csv(
                self.file, header=None, sep=' ',
                names=names, usecols=usecols,
                dtype={self.timeid: str}
            )
            .assign(
                timestamp=lambda x: pd.to_datetime(x[self.timeid], format="%Y%m%d%H%M%S")
            )
            .sort_values([self.userid, self.timeid])
        )
        return data
    

    def preprocess(self, data, window="1s"):

        return (
            data
            .pipe(drop_consequtive_repeats, window=window) ####### here !!!!
            .pipe(assign_session_id, session_break_delta=self.session_break_delta)
            .assign(
                sessid_global = lambda x: (x['timestamp'].diff() > pd.Timedelta(self.session_break_delta)).cumsum()
            )
        )

    def drop_conseq_repeats(self, data, window="1s"):
        return data.pipe(drop_consequtive_repeats, window=window)
    

    def create_sessid_column(self, data):
        return (
            data
            .pipe(assign_session_id, session_break_delta=self.session_break_delta)
            .assign(
                sessid_global = lambda x: (x['timestamp'].diff() > pd.Timedelta(self.session_break_delta)).cumsum()
            )
        )


    def _time_split_mask(self, cond):
        '''Split by time. Condition is checked on entire sessions, i.e.,
        it's met only if all session elements satisfy condition.'''
        def splitter(df):
            return cond(df['timestamp']).groupby([df['userid'], df['sessid']]).transform('all')
        return splitter


    def _idx2ind(self, data):
        """
            To fill in!
        """
        indices = uind, iind = [idx.get_indexer_for(data[idx.name]) for idx in self.indexers]
        dropped_inds = (uind == -1) | (iind == -1) # useful to keep for e.g. cold-start experiments
        checked_inds = ~dropped_inds
        reindexed_data = (
            data
            .loc[checked_inds]
            .assign(**{
                idx.name: ind[checked_inds] for ind, idx in zip(indices, self.indexers)
            })
        )
        return reindexed_data, dropped_inds


    def train_valid_test(self, data, reindex=True, test_interval='1d', valid_interval='1d'):
        """
            To fill in!
        """
        test_start_time = data[self.timeid].max() - pd.Timedelta(test_interval)
        valid_start_time = test_start_time - pd.Timedelta(valid_interval)

        train_data = data.loc[self._time_split_mask(lambda x: x < valid_start_time)]
        test_data = data.loc[self._time_split_mask(lambda x: x >= test_start_time)]
        valid_data = data.loc[self._time_split_mask(lambda x: (x >= valid_start_time) & (x < test_start_time))]

        uidx_cat = train_data[self.userid].astype('category').cat
        iidx_cat = train_data[self.itemid].astype('category').cat
        self.indexers = [
            uidx_cat.categories.rename(self.userid),
            iidx_cat.categories.rename(self.itemid)
        ]
        self.n_users, self.n_items = map(len, self.indexers)

        if reindex:
            new_indices = {
                self.userid: uidx_cat.codes,
                self.itemid: iidx_cat.codes
            }
            train_data = train_data.assign(**new_indices)
            valid_data, valid_dropped = self._idx2ind(valid_data)
            test_data, test_dropped = self._idx2ind(test_data)
        return train_data, valid_data, test_data


    def _freq_weight(self, x, coef=1, denom=1):
        return 1 + coef * np.log1p(x/denom)

    """
    def get_cui(self, data, coef=1, denom=1):
        
         #   data is assumed to be already reindexed!
        
        shape = (self.n_users, self.n_items)
        rows, cols = data[self.userid], data[self.itemid]
        freq_matrix = coo_matrix((np.ones_like(rows), (rows, cols)), shape=shape).tocsr()
        return freq_matrix._with_data(self._freq_weight(freq_matrix.data, coef, denom))
    """

    def get_freqs(self, data, c0=1, gamma=1):
        """
            data is assumed to be already reindexed!
        """
        shape = (self.n_users, self.n_items)
        rows, cols = data[self.userid], data[self.itemid]
        freq_matrix = (
            coo_matrix((np.ones_like(rows), (rows, cols)), shape=shape)
            .tocsr()
            .power(gamma)
        )
        freq_matrix = csr_matrix(c0 * freq_matrix / freq_matrix.sum(axis=1))
        return freq_matrix


    def get_seen_interactions(self, data):
        """
            data is assumed to be already reindexed!
        """
        shape = (self.n_users, self.n_items)
        rows, cols = data[self.userid], data[self.itemid]
        seen = (coo_matrix((np.ones_like(rows), (rows, cols)), shape=shape) > 0).astype(np.int32)
        return seen


    def get_users_sui(self, data, level=None):
        """
        `data` is assumed to be already reindexed.
        """
        kwargs = dict(
            shape = (self.n_items,)*2,
            userid = self.userid,
            itemid = self.itemid,
            exclude_item = dataprep.terminal_item
        )
        return generate_transition_matrices(data, level=level, **kwargs)
        
    
    def full_data_process(self, usecols, test_interval='1d'):
        
        # Read the data and sort by "userid" and "timestamp"
        self.data = self.read_data(usecols=usecols)

        # Drop consecutive repeats of apps and create "sessid" column
        self.data = self.preprocess(self.data)

        # Divide data into train/validation/test and transform user/item id to internal indexes
        self.train, self.valid, self.test = self.train_valid_test(self.data, reindex=True, test_interval=test_interval)

        self.seen_interactions = self.get_seen_interactions(self.train)

        self.train_sessions = list_session_lists(self.train)
        self.valid_sessions =  list_session_lists(self.valid, min_length=2)
        self.test_sessions =  list_session_lists(self.test, min_length=2)
    

    def prepare_data(self, usecols, test_interval, valid_interval, window=0, min_sess_length=2):
        # Read the data and sort by "userid" and "timestamp"
        self.data = self.read_data(usecols=usecols)
        if window != 0:
            self.data = self.drop_conseq_repeats(self.data, window=window)
        # Create "sessid" column
        self.data = self.create_sessid_column(self.data)

        # Divide data into train/validation/test and transform user/item id to internal indexes
        self.train, self.valid, self.test = self.train_valid_test(
            self.data,
            reindex=True,
            test_interval=test_interval,
            valid_interval=valid_interval,
        )

        self.seen_interactions = self.get_seen_interactions(self.train)

        self.train_sessions = list_session_lists(self.train)
        self.valid_sessions =  list_session_lists(
            self.valid, min_length=min_sess_length
        )
        self.test_sessions =  list_session_lists(
            self.test, min_length=min_sess_length
        )


    def _select_previous_items(self, data, level, k_step, fv=-1):
        result = pd.DataFrame()
        for k in range(1, k_step + 1):
            prev_items_data = (
                data[self.itemid]
                .shift(k, fill_value=fv)
            )
            prev_items_data.loc[data[level].diff()!=0] = fv # exclude inter-level connections
            result[k-1] = prev_items_data
        return result 


    def _counters2csr(self, df, k_step, len_group):
        rows = []
        cols = []
        vals = []
        for k, cr in enumerate(df):
            for n in df[k]:
                if n != -1:
                    rows.append(k)
                    cols.append(n)
                    vals.append(cr[n])
        item_csr = csr_matrix((vals, (rows, cols)), shape=(k_step, self.n_items)) / len_group
        return item_csr
    

    def get_sui_k_step(self, data, k_step, level):

        k_cols = [i for i in range(k_step)]

        result_dict = {}
        grouped_user_sess = data.groupby([self.userid, level])
        for name_us, group_us in grouped_user_sess:
            res_prev = self._select_previous_items(group_us, level, k_step)
            res_prev[self.userid] = group_us[self.userid]
            res_prev[self.itemid] = group_us[self.itemid]
            res_prev[level] = group_us[level]

            grouped_item = res_prev[1:].groupby('appid')# the first one will have -1, -1, -1 as no previous items
            i_dict = {}
            for name_i, group_i in grouped_item:
                len_group = len(group_i)
                counters = group_i[k_cols].apply(lambda x: Counter(x), axis=0)
                i_dict[name_i] = self._counters2csr(counters, k_step, len_group)
            result_dict[name_us] = i_dict
            
        new_dict = defaultdict(dict)
        for key in result_dict:
            new_dict[key[0]][key[1]] = result_dict[key]
        return new_dict