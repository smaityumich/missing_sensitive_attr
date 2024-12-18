import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.utils import resample
from numpy.random import RandomState


def get_random_seed(seed, ITER = 10):
    for _ in range(ITER):
        prng = RandomState(seed)
        seed = prng.randint(1e8)


def tranform_onehot(a):
    if len(a.shape) == 1:
        a = a.reshape((-1, 1))
    uniques = np.unique(a, axis = 0)
    onehots = [np.all(a == u.reshape((1, -1)), axis=1).astype('float') for u in uniques]    
    onehots = np.array(onehots).T
    return onehots

def predict_proba(X, estimators_, weights = None):
    if type(weights) == type(None):
        weights = np.ones(shape = (len(estimators_), ))
    weights = weights[:len(estimators_)]
    weights = weights.reshape((-1, 1, 1))
    probas = np.array([estimator.predict_proba(X) for estimator in estimators_])
    avg_probas = np.mean(probas * weights, axis=0) / np.mean(weights, axis = 0)
    return avg_probas

def predict(X, estimators_, weights = None):
    if type(weights) == type(None):
        weights = np.ones(shape = (len(estimators_), ))
    weights = weights[:len(estimators_)]
    weights = weights.reshape((-1, 1))
    predictions =  2 * np.array([estimator.predict(X) for estimator in estimators_]) - 1
    predictions = predictions * weights
    # Use majority voting for final prediction
    majority_votes = (predictions.sum(axis = 0) > 0).astype('float')
    return majority_votes



def constraint_weights_DP(conditions, sample_weight = None):
    
    if type(sample_weight) == type(None):
        sample_weight = np.ones(shape = (conditions.shape[0], ))

    # weights for constraints    

    if len(conditions.shape) == 1:
        conditions = conditions.reshape((-1, 1))
    uniques = np.unique(conditions, axis = 0)
    n_uniques = uniques.shape[0]


    A = - np.ones(shape = (n_uniques, n_uniques), dtype = 'float') / (n_uniques - 1)
    for i in range(n_uniques):
        A[i, i] = 1

    onehots = [np.all(conditions == u.reshape((1, -1)), axis=1).astype('float') for u in uniques]    
    onehots = np.array(onehots).T * sample_weight.reshape((-1, 1))
    weights = onehots / onehots.mean(axis = 0, keepdims = True)
    weights = weights @ A.T
    return weights

def constraint_weights_EO(conditions, y, sample_weight = None):
    
    if type(sample_weight) == type(None):
        sample_weight = np.ones(shape = (conditions.shape[0], ))
        
    # weights for constraints    

    if len(conditions.shape) == 1:
        conditions = conditions.reshape((-1, 1))

    if len(y.shape) == 1:
        y_mod = y.reshape((-1, 1))
        
    conditions_append = np.concatenate((conditions, y_mod), axis = 1)
    uniques = np.unique(conditions_append, axis = 0)
    n_uniques = uniques.shape[0]
    y_uniques = np.unique(y_mod, axis = 0).shape[0]
    conditions_uniques = np.unique(conditions, axis = 0).shape[0]

    assert n_uniques == y_uniques * conditions_uniques

        
    A = - np.ones(shape = (conditions_uniques, conditions_uniques), dtype = 'float') / (conditions_uniques - 1)
    for i in range(conditions_uniques):
        A[i, i] = 1
        
    A = np.kron(A, np.eye(y_uniques))

    # A_y = - np.ones(shape = (y_uniques, y_uniques), dtype = 'float') / (y_uniques - 1)
    # for i in range(y_uniques):
    #     A_y[i, i] = 1
        
    # A = np.kron(np.eye(conditions_uniques), A_y)

    onehots = [np.all(conditions_append == u.reshape((1, -1)), axis=1).astype('float') for u in uniques]    
    onehots = np.array(onehots).T  * sample_weight.reshape((-1, 1))
    weights = onehots / onehots.mean(axis = 0, keepdims = True)
    weights = weights @ A

    return weights

def constraint_weights_DP_v2(protected_onehots, sample_weight = None):
    
    n_uniques = protected_onehots.shape[1]
    A = - np.ones(shape = (n_uniques, n_uniques), dtype = 'float') / (n_uniques - 1)
    for i in range(n_uniques):
        A[i, i] = 1

    protected_onehots = protected_onehots * sample_weight.reshape((-1, 1))
    weights = protected_onehots / protected_onehots.mean(axis = 0, keepdims = True)
    weights = weights @ A.T
    return weights

def constraint_weights_EO_v2(protected_onehots, y, sample_weight = None):
    n_portected = protected_onehots.shape[1]
    y_onehot = tranform_onehot(y)
    n_y = y_onehot.shape[1]

    onehots = []
    for i_p in range(protected_onehots.shape[1]):
        for i_y in range(y_onehot.shape[1]):
            onehots.append(protected_onehots[:, [i_p]] * y_onehot[:, [i_y]])
    onehots = np.concatenate(onehots, axis = 1)
        
    A = - np.ones(shape = (n_portected, n_portected), dtype = 'float') / (n_portected - 1)
    for i in range(n_portected):
        A[i, i] = 1
    A = np.kron(A, np.eye(n_y))

    onehots = onehots  * sample_weight.reshape((-1, 1))
    weights = onehots / onehots.mean(axis = 0, keepdims = True)
    weights = weights @ A

    return weights


def constraint_weights(conditions, y = None, sample_weight = None):
    if type(y) == type(None):
        return constraint_weights_DP_v2(tranform_onehot(conditions), sample_weight)
    else:
        return constraint_weights_EO_v2(tranform_onehot(conditions), y, sample_weight)
    
def constraint_weights_v2(conditions, y = None, sample_weight = None):
    if type(y) == type(None):
        return constraint_weights_DP_v2(conditions, sample_weight)
    else:
        return constraint_weights_EO_v2(conditions, y, sample_weight)

def get_sample_weights(weights, lagrangian_constants, balancing = None):
    if type(balancing) == type(None):
        balancing = np.ones(shape = (weights.shape[0], ))
    sample_weight = balancing + (weights @ lagrangian_constants.reshape((-1, 1))).reshape((-1, ))
    return sample_weight


def get_lagrangian_constant(theta, B):
    l = np.exp(theta - theta.max())
    l = B * l / l.sum()
    return l[:-1]

def get_group_wise_loss(X, y, estimators_, weights, version = "v1"):
    assert version in ["v1", "v2"]
    if version == "v1":
        p = estimators_[-1].predict(X)
    else:
        p = predict(X, estimators_)
    loss = (p != y).astype('float')
    group_wise_loss = (weights.T @ loss.reshape((-1, 1)) / loss.shape[0])
    group_wise_loss = group_wise_loss.reshape((-1, ))
    return group_wise_loss

    

def update_theta(X, y, estimators_, lagrangian_constants_, weights, eps, version = "v1"):
    group_wise_loss = get_group_wise_loss(X, y, estimators_, weights, version=version)
    theta_update = group_wise_loss - eps
    return theta_update
    
def best_lagrangian(group_wise_loss, eps):
    theta_update = group_wise_loss - eps
    lambda_best = np.zeros_like(group_wise_loss, dtype = "float32")
    if theta_update.max() > 0:
        lambda_best[theta_update.argmax()] = 1
        return lambda_best
    else:
        return lambda_best


def reduction_update(X, y, max_iter, n_estimators, base_estimator,
                      random_state, weights, balancing, eps,
                      eta, B, verbose):
    tree_weights = np.zeros(shape = (max_iter, ))
    estimators_ = []
    flag = 0
    step = 0
    theta = np.zeros(shape = (weights.shape[1] + 1, ), dtype = 'float')
    lagrangian_constants_ = [get_lagrangian_constant(theta, B), ] 
    while (step < max_iter) and (flag < n_estimators):
        
        estimator = clone(base_estimator)
        random_state = get_random_seed(random_state)
        estimator.set_params(random_state = random_state)
        W = get_sample_weights(weights, lagrangian_constants_[-1], balancing = balancing.reshape((-1, )))
        random_state = get_random_seed(random_state)
        X_sample, y_sample, W_resample = resample(X, y, W, random_state = random_state)
        estimator.fit(X_sample, y_sample, sample_weight=W_resample)
        estimators_.append(estimator)
        group_wise_loss = get_group_wise_loss(X, y, estimators_, weights)
        t_update = group_wise_loss - eps
        if t_update.max() <= 0:
            flag += 1
            tree_weights[step] = 1
        theta[:-1] += eta * update_theta(X, y, estimators_, lagrangian_constants_, weights, eps)
        lagrangian_constants_.append(get_lagrangian_constant(theta, B))
        if verbose > 0:
            print("step {}".format(step), "n_estimators: {}".format(tree_weights.sum()))
        if verbose > 1:
            print("Step {}".format(step), (group_wise_loss).round(3), lagrangian_constants_[-1].round(3))
        step += 1
    
    tree_weights = tree_weights[:len(estimators_)]
    return estimators_, tree_weights

def reduction_update_v2(X, y, max_iter, n_estimators, base_estimator,
                      random_state, weights, balancing, eps,
                      eta, B, verbose):
    tree_current = np.zeros(shape = (max_iter, ))
    tree_index_ = []
    estimators_ = []
    flag = 0
    step = 0
    theta = np.zeros(shape = (weights.shape[1] + 1, ), dtype = 'float')
    lagrangian_constants_ = [get_lagrangian_constant(theta, B), ] 
    while (step < max_iter) and (flag < n_estimators):
        
        estimator = clone(base_estimator)
        random_state = get_random_seed(random_state)
        estimator.set_params(random_state = random_state)
        W = get_sample_weights(weights, lagrangian_constants_[-1], balancing = balancing.reshape((-1, )))
        random_state = get_random_seed(random_state)
        X_sample, y_sample, W_resample = resample(X, y, W, random_state = random_state)
        estimator.fit(X_sample, y_sample, sample_weight=W_resample)
        estimators_.append(estimator)
        tree_current[step] = 1
        group_wise_loss = get_group_wise_loss(X, y, estimators_, weights, version="v2")
        t_update = group_wise_loss - eps
        if t_update.max() <= 0:
            flag += 1
            tree_index_.append(np.copy(tree_current))
        theta[:-1] += eta * update_theta(X, y, estimators_, lagrangian_constants_, weights, eps, version="v2")
        lagrangian_constants_.append(get_lagrangian_constant(theta, B))
        if verbose > 0:
            print("step {}".format(step), "n_estimators: {}".format(len(tree_index_)))
        if verbose > 1:
            print("Step {}".format(step), (group_wise_loss).round(3), lagrangian_constants_[-1].round(3))
        step += 1
    # tree_weights = np.array(tree_index_).T
    tree_weights = np.array(tree_index_).sum(axis = 0)
    tree_weights = tree_weights[:len(estimators_)]
    return estimators_, tree_weights
    


def reduction_exponential_gradient_demographi_parity(
        X, y, a, base_estimator = None, B = 2, eps = 0.02, n_estimators = 100,
        eta = 1, random_state = 42, 
        sample_weight = None, max_iter = 500, class_weight = "balanced", 
        verbose = 0, onehot_protected = False):
    
    if type(base_estimator) == type(None):
        base_estimator = DecisionTreeClassifier(max_depth=10, criterion="gini")
        
    assert class_weight in ["None", "balanced"]
    
    if class_weight == "None":
        balancing = np.ones_like(y)
    else:
        balancing = y / np.mean(y) + (1 - y) / np.mean(1 - y)
        
    if type(sample_weight) == type(None):
        sample_weight = np.ones_like(y)
    
    estimators_, estimators_final_ = [], []
    if onehot_protected:
        weights = constraint_weights_v2(conditions=a, sample_weight=sample_weight)
    else:
        weights = constraint_weights(conditions=a, sample_weight=sample_weight)
    theta = np.zeros(shape = (weights.shape[1] + 1, ), dtype = 'float')
    lagrangian_constants_ = [get_lagrangian_constant(theta, B), ] 
    
    
    
    flag = 0
    step = 0

    while (step < max_iter) and (flag < n_estimators):
        
        estimator = clone(base_estimator)
        random_state = get_random_seed(random_state)
        estimator.set_params(random_state = random_state)
        W = get_sample_weights(weights, lagrangian_constants_[-1], balancing = balancing.reshape((-1, )))
        random_state = get_random_seed(random_state)
        X_sample, y_sample, W_resample = resample(X, y, W, random_state = random_state)
        estimator.fit(X_sample, y_sample, sample_weight=W_resample)
        estimators_.append(estimator)
        group_wise_loss = get_group_wise_loss(X, y, estimators_, weights)
        t_update = group_wise_loss - eps
        if t_update.max() <= 0:
            flag += 1
            estimators_final_.append(estimator)
        theta[:-1] += eta * update_theta(X, y, estimators_, lagrangian_constants_, weights, eps)
        lagrangian_constants_.append(get_lagrangian_constant(theta, B))
        if verbose > 0:
            print("step {}".format(step), "n_estimators: {}".format(len(estimators_final_)))
        if verbose > 1:
            print("Step {}".format(step), (group_wise_loss).round(3), lagrangian_constants_[-1].round(3))
        step += 1
    return estimators_final_


    

def reduction_exponential_gradient(
        X, y, a, base_estimator = None, B = 1, eps = 0.02, n_estimators = 100,
        eta = 0.1, sample_weight = None, max_iter = 500, class_weight = "balanced", 
        verbose = 0, onehot_protected = False, random_state = 42, constraint = "EO"):
    
    
    """
    X: numpy array of covariates
    
    y: numpy array of responses
    
    a: protected attributes; onehot encoded if onehot_protected = True
    
    base_estimator: sklearn estimator for ensemble 
    
    B: (float) controls magnitude of Lagrangian, higher B -> higher lagrangian -> stricter constraint
    
    eps: constraint violation
    
    n_estimator: maximum number of sklearn estimator
    
    eta: step-size for Lagrangian update
    
    sample_weight: sample weights for risk minimization
    
    max_iter: maximum number of iteration
    
    class_weight: either "balanced" or "None"
    
    verbose: 0, 1, 2 depending on the iteration info
    
    onehot_protected: whether to provide onehot encoding for protected attribute
    
    random_state: random seed for reproducibility
    
    constraint: either demographi parity "DP" or equalized odds "EO"; default "EO"
    
    """
    
    
    if type(base_estimator) == type(None):
        base_estimator = DecisionTreeClassifier(max_depth=10, criterion="gini")
        
    assert class_weight in ["None", "balanced"]
    
    if class_weight == "None":
        balancing = np.ones_like(y)
    else:
        balancing = y / np.mean(y) + (1 - y) / np.mean(1 - y)
        
    if type(sample_weight) == type(None):
        sample_weight = np.ones_like(y)
    
    assert constraint in ["DP", "EO"]
    
    if constraint == "DP":
        y_temp = None
    else:
        y_temp = y
    
    if onehot_protected:
        weights = constraint_weights_v2(conditions=a, y = y_temp, sample_weight=sample_weight)
    else:
        weights = constraint_weights(conditions=a, y = y_temp, sample_weight=sample_weight)
    
    return reduction_update(X, y, max_iter, n_estimators, base_estimator,
                          random_state, weights, balancing, eps,
                          eta, B, verbose)

def reduction_exponential_gradient_v2(
        X, y, a, base_estimator = None, B = 1, eps = 0.02, n_estimators = 100,
        eta = 0.1, sample_weight = None, max_iter = 500, class_weight = "balanced", 
        verbose = 0, onehot_protected = False, random_state = 42, constraint = "EO"):
    
    
    """
    X: numpy array of covariates
    
    y: numpy array of responses
    
    a: protected attributes; onehot encoded if onehot_protected = True
    
    base_estimator: sklearn estimator for ensemble 
    
    B: (float) controls magnitude of Lagrangian, higher B -> higher lagrangian -> stricter constraint
    
    eps: constraint violation
    
    n_estimator: maximum number of sklearn estimator
    
    eta: step-size for Lagrangian update
    
    sample_weight: sample weights for risk minimization
    
    max_iter: maximum number of iteration
    
    class_weight: either "balanced" or "None"
    
    verbose: 0, 1, 2 depending on the iteration info
    
    onehot_protected: whether to provide onehot encoding for protected attribute
    
    random_state: random seed for reproducibility
    
    constraint: either demographi parity "DP" or equalized odds "EO"; default "EO"
    
    """
    
    
    if type(base_estimator) == type(None):
        base_estimator = DecisionTreeClassifier(max_depth=10, criterion="gini")
        
    assert class_weight in ["None", "balanced"]
    
    if class_weight == "None":
        balancing = np.ones_like(y)
    else:
        balancing = y / np.mean(y) + (1 - y) / np.mean(1 - y)
        
    if type(sample_weight) == type(None):
        sample_weight = np.ones_like(y)
    
    assert constraint in ["DP", "EO"]
    
    if constraint == "DP":
        y_temp = None
    else:
        y_temp = y
    
    if onehot_protected:
        weights = constraint_weights_v2(conditions=a, y = y_temp, sample_weight=sample_weight)
    else:
        weights = constraint_weights(conditions=a, y = y_temp, sample_weight=sample_weight)
    
    return reduction_update_v2(X, y, max_iter, n_estimators, base_estimator,
                          random_state, weights, balancing, eps,
                          eta, B, verbose)

def reduction_exponential_gradient_equalized_odds(
        X, y, a, base_estimator = None, B = 1, eps = 0.02, n_estimators = 100,
        eta = 0.1, sample_weight = None, max_iter = 500, class_weight = "balanced", 
        verbose = 0, onehot_protected = False, random_state = 42):
    
    
    """
    X: numpy array of covariates
    
    y: numpy array of responses
    
    a: protected attributes; onehot encoded if onehot_protected = True
    
    base_estimator: sklearn estimator for ensemble 
    
    B: (float) controls magnitude of Lagrangian, higher B -> higher lagrangian -> stricter constraint
    
    eps: constraint violation
    
    n_estimator: maximum number of sklearn estimator
    
    eta: step-size for Lagrangian update
    
    sample_weight: sample weights for risk minimization
    
    max_iter: maximum number of iteration
    
    class_weight: either "balanced" or "None"
    
    verbose: 0, 1, 2 depending on the iteration info
    
    onehot_protected: whether to provide onehot encoding for protected attribute
    
    random_state: random seed for reproducibility
    
    """
    
    
    if type(base_estimator) == type(None):
        base_estimator = DecisionTreeClassifier(max_depth=10, criterion="gini")
        
    assert class_weight in ["None", "balanced"]
    
    if class_weight == "None":
        balancing = np.ones_like(y)
    else:
        balancing = y / np.mean(y) + (1 - y) / np.mean(1 - y)
        
    if type(sample_weight) == type(None):
        sample_weight = np.ones_like(y)
    
    estimators_, estimators_final_ = [], []
    
    if onehot_protected:
        weights = constraint_weights_v2(conditions=a, y = y, sample_weight=sample_weight)
    else:
        weights = constraint_weights(conditions=a, y = y, sample_weight=sample_weight)
    theta = np.zeros(shape = (weights.shape[1] + 1, ), dtype = 'float')
    lagrangian_constants_ = [get_lagrangian_constant(theta, B), ] 
    
    
    
    flag = 0
    step = 0

    while (step < max_iter) and (flag < n_estimators):
        
        ## Clone estimator and set random state
        estimator = clone(base_estimator)
        random_state = get_random_seed(random_state)
        estimator.set_params(random_state = random_state)
        
        ## get per-sample weights for Lagrangian loss
        W = get_sample_weights(weights, lagrangian_constants_[-1], balancing = balancing.reshape((-1, )))
        
        ## rasample for tree bagging
        random_state = get_random_seed(random_state)
        X_sample, y_sample, W_resample = resample(X, y, W, random_state = random_state)
        
        
        estimator.fit(X_sample, y_sample, sample_weight=W_resample)
        estimators_.append(estimator)
        group_wise_loss = get_group_wise_loss(X, y, estimators_, weights)
        t_update = group_wise_loss - eps
        if t_update.max() <= 0:
            flag += 1
            estimators_final_.append(estimator)
        theta[:-1] += eta * update_theta(X, y, estimators_, lagrangian_constants_, weights, eps) #* ((step + 1) ** (-1/3))
        lagrangian_constants_.append(get_lagrangian_constant(theta, B))
        if verbose > 0:
            print("step {}".format(step), "n_estimators: {}".format(len(estimators_final_)))
        if verbose > 1:
            print("Step {}".format(step), (group_wise_loss).round(3), lagrangian_constants_[-1].round(3))
        step += 1
    print("Number of trees", len(estimators_final_))
    return estimators_final_