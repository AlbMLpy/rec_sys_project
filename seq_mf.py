import numpy as np
from scipy.sparse import csr_matrix

local_factors_names = ['long-term', 'contextual']
global_factors_names = ['items', 'contexts']

# MODEL: SeqMF with eta that is a number for every user (2.3.1), no sessions

def prepare_target_update(factors, target_factor_id, update_funcs, updater, *args):
    target_factors = factors[target_factor_id]
    update_func = update_funcs[target_factor_id]
    target_update_func = updater(update_func, *args)
    return target_factors, target_update_func

# Server side:
def update_global_model(
    confidence, transitions,
    local_factors, global_factors,
    gradient_funcs, regularization, gain, n_iter,
    evaluation_callback, iterator,  ### TO DEBUG
):
    """
        To fill in!
    """
    iterator = iterator or range
    for _ in iterator(n_iter):
        for factor_name in global_factors_names:
            update_global_factor(
                confidence, transitions,
                local_factors, global_factors, factor_name,
                gradient_funcs, regularization, gain
            )
        if evaluation_callback:
            evaluation_callback(local_factors, global_factors)
    


def update_global_factor(
    confidence, transitions,
    local_factors, global_factors, factor_name,
    gradient_funcs, regularization, gain
):
    target_factor_id = global_factors_names.index(factor_name)
    updater_args = (  # will go into `fetch_items_grad` or `fetch_context_grad`
        confidence, transitions, local_factors, global_factors
    )
    target_factor, gradient_update = prepare_target_update(
        global_factors, target_factor_id, gradient_funcs, gradient_updater, *updater_args
    )

    grad = regularization * target_factor
    n_users, _ = confidence.shape
    for user in range(n_users):
        grad += gradient_update(user)
    target_factor -= gain * grad


def gradient_updater(gradient_func, confidence, transitions, local_factors, global_factors):
    Q, Qb = global_factors
    empty_Su = csr_matrix((Q.shape[0], Qb.shape[0]))
    grad_params = () # for later use with e.g. Adam
    def updater(user):
        item_transitions = transitions.get(user, empty_Su) # TODO: no need for update if user_Su's empty in the context factor case
        return gradient_func(
            confidence, item_transitions,
            *local_factors, *global_factors,
            user, *grad_params
        )
    return updater


def _compute_user_preference(Su, P, eta, Q, Qb, u):
    SuQb = (Su @ Qb)
    #seq_part = np.einsum('ij,ij->i', SuQb, Q, optimize=False)
    seq_part = (SuQb * Q).sum(axis=1)
    if eta is not None:
        seq_part *= eta[u]
    r_u = np.dot(Q, P[u]) + seq_part
    return r_u, SuQb


def _update_weighted_error(Cui, r, u):
    indptr = Cui.indptr
    inds = Cui.indices[indptr[u]:indptr[u + 1]]
    coef = Cui.data[indptr[u]:indptr[u + 1]]
    r[inds] = (r[inds] - 1.0) * coef


def fetch_items_grad(Cui, Su, P, eta, Q, Qb, u):
    r_u, SuQb = _compute_user_preference(Su, P, eta, Q, Qb, u)
    _update_weighted_error(Cui, r_u, u)
    longterm_grad = np.outer(r_u, P[u])
    sessions_grad = r_u[:, np.newaxis] * SuQb
    if eta is not None:
        return longterm_grad + eta[u] * sessions_grad
    return longterm_grad + sessions_grad


def fetch_context_grad(Cui, Su, P, eta, Q, Qb, u):
    r_u, _ = _compute_user_preference(Su, P, eta, Q, Qb, u)
    _update_weighted_error(Cui, r_u, u)
    sessions_grad = Su.T @ (r_u[:, np.newaxis] * Q)
    if eta is not None:
        return eta[u] * sessions_grad
    return sessions_grad


# Client side:
def update_local_models(
    confidence, transitions,
    local_factors, global_factors,
    least_squares_funcs, regularization
):
    for factor_name in local_factors_names:
        if local_factors[local_factors_names.index(factor_name)] is None:
            continue
        update_local_factor(
            confidence, transitions,
            local_factors, factor_name, global_factors,
            least_squares_funcs, regularization
        )

def least_squares_updater(least_squares_func, confidence, transitions, other, global_factors, regularization):
    Q, Qb = global_factors
    QtQ = np.zeros(shape=(Q.shape[1], Q.shape[1])) #TO FIX!!!!
    #QtQ = Q.T @ Q
    QtQ[np.diag_indices_from(QtQ)] += regularization #TO FIX!!!!
    empty_Su = csr_matrix((Q.shape[0], Qb.shape[0]))
    def updater(user):
        item_transitions = transitions.get(user, empty_Su) # TODO: no need for update if user_Su's empty in the context factor case
        return least_squares_func(
            confidence, item_transitions,
            other, *global_factors, QtQ, user, regularization
        )
    return updater      

def update_local_factor(
    confidence, transitions,
    local_factors, factor_name, global_factors,
    least_squares_funcs, regularization
):
    '''
    Implicitly assumes there're only 2 types of user factors.
    '''
    target_factor_id = local_factors_names.index(factor_name)
    other_factor_id = 1 - target_factor_id # will break if there're >2 types of factors
    updater_args = ( # will go into `least_squares_long_term` or `least_squares_contextual`
        confidence, transitions,
        local_factors[other_factor_id], global_factors, regularization
    )
    target_factor, least_squares_update = prepare_target_update(
        local_factors, target_factor_id,
        least_squares_funcs, least_squares_updater, *updater_args
    )
    n_users, _ = confidence.shape
    for u in range(n_users):
        target_factor[u] = least_squares_update(u) # each client does independently


def least_squares_long_term(Cui, Su, eta, Q, Qb, QtQ, u, reg):
    A, b = user_linear_equation(Cui, Su, eta, Q, Qb, QtQ, u)
    return np.linalg.solve(A, b)

def least_squares_contextual(Cui, Su, P, Q, Qb, QtQ, u, reg):
    indptr = Cui.indptr
    inds = Cui.indices[indptr[u]:indptr[u + 1]]
    coef = Cui.data[indptr[u]:indptr[u + 1]]

    long_term_err = (1.0 - np.dot(Q[inds], P[u]))
    seq_part = np.einsum('ij,ij->i', Su@Qb, Q, optimize=False)
    weighted_seq_part = coef * seq_part[inds]
    denom = np.dot(seq_part[inds], weighted_seq_part)
    return np.dot(long_term_err, weighted_seq_part) / (denom + reg)


def user_linear_equation(Cui, Su, eta, Q, Qb, QtQ, u):
    indptr = Cui.indptr
    inds = Cui.indices[indptr[u]:indptr[u + 1]]
    coef = Cui.data[indptr[u]:indptr[u + 1]] 
    
    # calculate A
    Qnnz = Q[inds]
    #A = QtQ + (Qnnz.T * (coef - 1)) @ Qnnz
    A = QtQ + (Qnnz.T * coef) @ Qnnz

    # calculate b
    seq_part = -np.einsum('ij,ij->i', Su@Qb, Q, optimize=False)
    if eta is not None:
        seq_part *= eta[u]
    seq_part[inds] = (seq_part[inds] + 1.0) * coef  
    b = Q.T @ seq_part
    return A, b  


def initialize_model(n_users, n_items, n_factors, seed, use_eta=False):
    random_state = np.random.RandomState(seed)
    user_factors_long_term = random_state.normal(0, 0.01, size=(n_users, n_factors))
    user_factors_contextual = None
    if use_eta:
        user_factors_contextual = random_state.normal(0, 0.01, size=(n_users,))
    local_factors = [user_factors_long_term, user_factors_contextual] # order must comply with local_factors_names
    least_squares = [least_squares_long_term, least_squares_contextual] # order must comply with local_factors_names
    
    item_factors = random_state.normal(0, 0.01, size=(n_items, n_factors))
    context_factors = random_state.normal(0, 0.01, size=(n_items, n_factors))
    global_factors = [item_factors, context_factors] # order must comply with global_factors_names
    gradient_funcs = [fetch_items_grad, fetch_context_grad] # order must comply with global_factors_names
    return local_factors, global_factors, least_squares, gradient_funcs


# Model training functionality:
def fit_model(
    confidence_matrix, transition_matrices, n_factors, regularization,
    gain, *, n_epochs=1, n_iter=20, seed=42, evaluation_callback=None,
    iterator=None, use_eta=False,
):
    """
        To fill in!
    """
    n_users, n_items = confidence_matrix.shape
    local_factors, global_factors, least_squares, gradient_funcs = initialize_model(
        n_users, n_items, n_factors, seed, use_eta,
    )
    common_args = dict(
        confidence = confidence_matrix,
        transitions = transition_matrices,
        local_factors = local_factors,
        global_factors = global_factors,
        regularization=regularization,
    )
    iterator = iterator or range
    for epoch in iterator(n_epochs):
        update_global_model(
            **common_args,
            gradient_funcs=gradient_funcs,
            gain=gain,
            n_iter=n_iter,
        )

        update_local_models(
            **common_args,
            least_squares_funcs=least_squares,
        )

        if evaluation_callback:
            evaluation_callback(local_factors, global_factors)
        
    return local_factors, global_factors


def fit_restricted(
    confidence_matrix, transition_matrices, n_factors,
    regularization, gain, *, n_iter=1, seed=42,
    evaluation_callback=None, use_eta=False, iterator=None,
    **kwargs # compatibility
):
    n_users, n_items = confidence_matrix.shape
    local_factors, global_factors, least_squares, gradient_funcs = initialize_model(
        n_users, n_items, n_factors, seed, use_eta,
    )
    common_args = dict(
        confidence=confidence_matrix,
        transitions=transition_matrices,
        local_factors=local_factors,
        global_factors=global_factors,
        regularization=regularization,
    )
    update_local_models(
        **common_args,
        least_squares_funcs=least_squares,
    )
    update_global_model(
        **common_args,
        gradient_funcs=gradient_funcs,
        gain=gain,
        n_iter=n_iter,
        evaluation_callback=evaluation_callback,
        iterator=iterator,

    )
    update_local_models(
        **common_args,
        least_squares_funcs=least_squares,
    )
    return local_factors, global_factors


def get_scores_generator(local_factors, global_factors):
    P, eta = local_factors
    Q, Qb = global_factors
    if eta is not None:
        def generate_scores(uid, sid, sess_items, item_pool):
            scores = (P[uid] + eta[uid] * Qb[sess_items[-1]]) @ Q[item_pool].T
            return scores
        return generate_scores

    def generate_scores(uid, sid, sess_items, item_pool):
        scores = (P[uid] + Qb[sess_items[-1]]) @ Q[item_pool].T
        return scores
    return generate_scores
    