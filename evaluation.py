from collections import Counter
from itertools import accumulate
from bisect import bisect_left
from math import log, log1p

import numpy as np
import pandas as pd

from lib import index, searchranked
ln2 = log(2)

def get_known_items(udx, seen_interactions):
    """.Checked."""
    indptr = seen_interactions.indptr
    indices = seen_interactions.indices    
    return indices[indptr[udx]:indptr[udx + 1]]


def find_target_rank(target_item, generate_scores, uid, sid, sess_items, item_pool, topk):
    """.Checked."""
    predict_for_items = item_pool
    target_pos = index(item_pool, target_item)
    if target_pos is None: # i.e., app is not installed
        # two possible cases:
        # - target item is known in general, but not installed on this device
        # - completely new app, not seen in training, i.e. no index in global model
        # both cases must be gracefully handled by a model
        target_pos = len(item_pool) // 2 # avoid boundary positions - helps catching bugs
        predict_for_items = np.r_[item_pool[:target_pos], target_item, item_pool[target_pos:]]
    scores = generate_scores(uid, sid, sess_items, predict_for_items)
    item_rank = searchranked(scores, target_pos, topk)
    return item_rank


def metric_increments(ranks, numk, topk_bins):
    '''
    Groups and accumulates metrics in bins corresponding to top-k intervals.
    E.g., for topk=[1,3,5] it will accumulate into intervals [1], [2-3], [4-5].
    Then, in order to obtain "<=k" intervals, one has to compute a running sum.
    Relies on ranks containing values not larger than max topk. 
    '''
    hits = [0] * numk
    reci = [0] * numk
    reci_ln1p = [0] * numk
    counter = Counter(ranks)
    counter.pop(0, None) # remove invalid rank if present
    for rank, freq in counter.items(): # groups of ranks and their frequencies
        k_bin = topk_bins[rank] # a bin the group belongs to based on topk interval
        hits[k_bin] += freq
        reci[k_bin] += freq / rank
        reci_ln1p[k_bin] += freq / log1p(rank) # log1p = ln(1+x)
    reci_log1p = [ln2*x for x in reci_ln1p] # log2(1+x) = log(1+x) / log(2)
    return hits, reci, reci_log1p


def incremental_metrics(topk_list):
    '''
    Note: map metric is not computed, as map equals mrr for 1 holdout item.
    example: https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank
    '''
    numk = len(topk_list)
    metrics = [hr, mrr, ndcg] = [[0]*numk, [0]*numk, [0]*numk]
    topk_bins = [bisect_left(topk_list, x) for x in range(topk_list[-1]+1)]

    def scorer(ranks):
        cnt = len(ranks)
        scorer.counts += cnt
        increments = metric_increments(ranks, numk, topk_bins)
        # accumulate average values incrementally
        hr[:], mrr[:], ndcg[:] = (
            [val + (inc-cnt*val) / scorer.counts for val, inc in zip(metric, increment)]
            for metric, increment in zip(metrics, map(accumulate, increments))
        )
    scorer.counts = 0
    return scorer, {'hr': hr, 'mrr': mrr, 'ndcg': ndcg}


def collect_metrics(generate_scores, test_sessions, seen_interactions, topk_list):
    """
        generate_scores: generate_scores(uid, sid, sess_items, predict_for_items) -> np.array;
        test_sessions: {uid1: [s1, s2, s3], ...}, s1 = [iid1, iid2, ...];
        seen_interactions: csr_matrix;
        topk_list: sorted list of top-k values;
    """
    # get the dictionary of item positions:
    user_metrics = {}
    user_stats = {}
    max_topk = topk_list[-1]

    sid_increment = 0
    for uid, sessions in test_sessions.items():
        user_stats[uid] = stats = []
        metrics_updater, metrics = incremental_metrics(topk_list)
        item_pool = get_known_items(uid, seen_interactions) # i.e., all installed apps on user device
        for sid, session in enumerate(sessions):
            sess_ranks = []
            sess_items = []
            for i in range(len(session)-1):
                sess_items.append(session[i])
                target_rank = find_target_rank(
                    session[i+1], generate_scores,
                    uid, sid_increment+sid, sess_items,
                    item_pool, max_topk
                )
                sess_ranks.append(target_rank)
            metrics_updater(sess_ranks)
            stats.append(sess_ranks)
        sid_increment += len(sessions)
        user_metrics[uid] = metrics
    return user_metrics, user_stats


def evaluate(generate_scores, test_sessions, seen_interactions, topk=(1, 3, 5)) -> pd.DataFrame:
    """
        generate_scores: generate_scores(???) -> np.array;
        test_sessions: {uid1: [s1, s2, s3], ...}, s1 = [iid1, iid2, ...];
        seen_interactions: csr_matrix;
        topk: list ot top-k values to evaluate against
    """
    topk = sorted(topk)
    user_metrics, user_stats = collect_metrics(generate_scores, test_sessions, seen_interactions, topk)
    metrics_df = pd.concat(
        {user: pd.DataFrame(metrics, index=topk) for user, metrics in user_metrics.items()},
        names=['userid', 'topk']
    ).rename_axis(columns='metrics')
    return metrics_df, user_stats


####### For ipynb experiments
def append_result(stat_dict, metrics, factor_norms):
    for metric in metrics.keys():
        for topk in metrics[metric].keys():
            stat_dict[metric][topk].append(metrics[metric][topk])
    
    for factor, norm in factor_norms.items():
        stat_dict["norms"][factor].append(norm)

def get_stat_dict(metric_names, topk, factor_names):
    stat = {metric: {k: [] for k in topk} for metric in metric_names}
    stat["norms"] = {name: [] for name in factor_names}
    return stat
 
def nm_zero(fact):
    if isinstance(fact, np.ndarray):
        return np.linalg.norm(fact)
    return -1

def evaluation_callback(
    get_scores_generator,
    test_sessions,
    seen_interactions,
    factor_names,
    rmse_loss=None,
    rmse_params=None, 
):
    """
        To fill in!
    """

    def callback(local_factors, global_factors):
        scorer = get_scores_generator(local_factors, global_factors)
        metrics_df, user_stats = evaluate(scorer, test_sessions, seen_interactions)
        factor_norms = {fname: nm_zero(fact) for fname, fact in zip(factor_names, local_factors + global_factors)}
        results = (
            metrics_df
            .reset_index()
            .groupby(["topk"])
            .mean()[["hr", "mrr", "ndcg"]].to_dict()
        )
        append_result(callback.stat, results, factor_norms)
        if rmse_loss:
            callback.rmse.append(rmse_loss(rmse_params["Cui"], *local_factors, *global_factors))
    
    callback.stat = get_stat_dict(
        metric_names=["hr", "mrr", "ndcg"],
        topk=[1, 3, 5],
        factor_names=factor_names
    )
    callback.rmse = []
    return callback
