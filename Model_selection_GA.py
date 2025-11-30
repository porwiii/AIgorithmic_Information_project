import numpy as np
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# ============================================
# 1. MDL — 2 versions
# ============================================
def mdl_bic(y_true, y_pred, k):
    n = len(y_true)
    mse = mean_squared_error(y_true, y_pred)
    mse = max(mse, 1e-12)
    return k * np.log(n) + n * np.log(mse)


def abs_app(M, eps=1e-6):
    M = np.asarray(M)
    return np.sqrt(M ** 2 + eps)


def Cmatrix(M):
    M = np.asarray(M)
    return np.sum(np.log2(1.0 + abs_app(M)))


# def Cmatrix_tolerant(M, eps=1.0):
#     M = np.asarray(M)
#     # small errors are trimmed to zero
#     M_eff = np.where(np.abs(M) < eps, 0.0, M)
#     return np.sum(np.log2(1.0 + abs_app(M_eff)))


def Cresiduals_with_min1(residuals, tol=0.1, eps=1e-6):
    """
    Error encoding:
    - |r| <= tol  → cost = 0 (we ignore the error)
    - |r| >  tol  → cost >= 1 bit (1 + log2(1 + |r|))
    """
    r = np.asarray(residuals)
    ax = np.sqrt(r ** 2 + eps)

    # mask: which residuals are “significant”
    important = ax > tol

    cost = np.zeros_like(ax, dtype=float)

    # for significant errors: 1 bit “signal” + logarithmic penalty
    cost[important] = 1.0 + np.log2(1.0 + ax[important])

    return np.sum(cost)


def Cparams_with_min1(params, tol=0.0, eps=1e-6):
    p = np.asarray(params)
    ap = np.sqrt(p ** 2 + eps)

    # non-zero parameters
    important = ap > tol

    cost = np.zeros_like(ap, dtype=float)
    cost[important] = 1.0 + np.log2(1.0 + ap[important])
    return np.sum(cost)


# ============================================
# 2. Model from an individual
# ============================================
MODEL_TYPES_ALL = ["linear", "poly", "spline", "tree", "ridge", "rf", "mlp"]

def build_model(individual):
    mtype = individual["model_type"]

    if mtype == "linear":
        # standard linear regression
        model = LinearRegression()

        def fit_predict(X, y):
            model.fit(X, y)
            y_pred = model.predict(X)
            # parameter vector: coefficients + intercept
            coef = np.ravel(model.coef_)
            intercept = np.atleast_1d(model.intercept_)
            params = np.concatenate([coef, intercept])
            return y_pred, params

    elif mtype == "poly":
        degree = individual["degree"]
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        lin = LinearRegression()

        def fit_predict(X, y):
            X_poly = poly.fit_transform(X)
            lin.fit(X_poly, y)
            y_pred = lin.predict(X_poly)
            coef = np.ravel(lin.coef_)
            intercept = np.atleast_1d(lin.intercept_)
            params = np.concatenate([coef, intercept])
            return y_pred, params

    elif mtype == "spline":
        n_knots = individual["n_knots"]
        spline = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            include_bias=True
        )
        lin = LinearRegression()

        def fit_predict(X, y):
            X_spl = spline.fit_transform(X)
            lin.fit(X_spl, y)
            y_pred = lin.predict(X_spl)
            coef = np.ravel(lin.coef_)
            intercept = np.atleast_1d(lin.intercept_)
            params = np.concatenate([coef, intercept])
            return y_pred, params

    elif mtype == "tree":
        max_depth = individual["max_depth"]
        tree = DecisionTreeRegressor(max_depth=max_depth)

        def fit_predict(X, y):
            tree.fit(X, y)
            y_pred = tree.predict(X)

            # take values of leaves as parameters
            tree_ = tree.tree_
            is_leaf = (tree_.children_left == -1)
            leaf_values = tree_.value[is_leaf, ...].ravel()
            params = leaf_values
            return y_pred, params

    # -------- NEW MODELS --------
    elif mtype == "ridge":
        # ridge regression: like linear but with regularization
        alpha = individual["alpha"]
        model = Ridge(alpha=alpha)

        def fit_predict(X, y):
            model.fit(X, y)
            y_pred = model.predict(X)
            coef = np.ravel(model.coef_)
            intercept = np.atleast_1d(model.intercept_)
            params = np.concatenate([coef, intercept])
            return y_pred, params

    elif mtype == "rf":
        # RandomForestRegressor — parameters are leaf values of all trees
        n_estimators = individual["n_estimators"]
        max_depth = individual["max_depth"]
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=0
        )

        def fit_predict(X, y):
            rf.fit(X, y)
            y_pred = rf.predict(X)
            leaf_params = []
            for est in rf.estimators_:
                tree_ = est.tree_
                is_leaf = (tree_.children_left == -1)
                leaf_vals = tree_.value[is_leaf, ...].ravel()
                leaf_params.append(leaf_vals)
            if leaf_params:
                params = np.concatenate(leaf_params)
            else:
                params = np.zeros(1)
            return y_pred, params

    elif mtype == "mlp":
        # small MLP neural network
        hidden = individual["hidden_size"]
        alpha = individual["alpha"]
        mlp = MLPRegressor(
            hidden_layer_sizes=(hidden,),
            activation="relu",
            alpha=alpha,
            max_iter=2000,
            random_state=0
        )

        def fit_predict(X, y):
            mlp.fit(X, y)
            y_pred = mlp.predict(X)
            # all weights and biases as parameters
            weight_vecs = [w.ravel() for w in mlp.coefs_]
            bias_vecs = [b.ravel() for b in mlp.intercepts_]
            params = np.concatenate(weight_vecs + bias_vecs)
            return y_pred, params

    else:
        raise ValueError(f"Unknown model_type: {mtype}")

    return fit_predict


def evaluate_individual(individual, X, y, mdl_type="bic"):
    fit_predict = build_model(individual)
    y_pred, params = fit_predict(X, y)

    if mdl_type == "bic":
        k = len(params)
        return mdl_bic(y, y_pred, k)

    elif mdl_type == "cmatrix":
        C_mod = Cmatrix(params) + np.log2(len(params))
        residuals = y - y_pred
        C_dat = Cmatrix(residuals)
        return C_mod + C_dat

    elif mdl_type == "cmatrix_with_min1":
        C_mod = Cparams_with_min1(params) + np.log2(len(params))
        residuals = y - y_pred
        C_dat = Cresiduals_with_min1(residuals)
        return C_mod + C_dat

    else:
        raise ValueError(f"Unknown mdl_type: {mdl_type}")


# ============================================
# 3. GA
# ============================================
def random_individual(
        max_poly_degree=8,
        min_knots=3,
        max_knots=10,
        max_depth=6,
        model_types=None,
):
    if model_types is None:
        model_types = MODEL_TYPES_ALL

    return {
        "model_type": random.choice(model_types),
        "degree": random.randint(2, max_poly_degree),
        "n_knots": random.randint(min_knots, max_knots),
        "max_depth": random.randint(1, max_depth),
        # new:
        "alpha": 10 ** random.uniform(-3, 3),
        "n_estimators": random.randint(10, 100),
        "hidden_size": random.randint(5, 50),
    }


def mutate_individual(
        individual,
        max_poly_degree=8,
        min_knots=3,
        max_knots=10,
        max_depth=6,
        min_alpha=1e-3,
        max_alpha=1e3,
        min_estimators=10,
        max_estimators=100,
        min_hidden=5,
        max_hidden=50,
        model_mut_prob=0.2,
        param_mut_prob=0.5,
        model_types=None,
):
    if model_types is None:
        model_types = MODEL_TYPES_ALL

    ind = individual.copy()

    # mutation of model type
    if random.random() < model_mut_prob:
        ind["model_type"] = random.choice(model_types)
        # when type changes, re-randomize parameters
        ind["degree"] = random.randint(2, max_poly_degree)
        ind["n_knots"] = random.randint(min_knots, max_knots)
        ind["max_depth"] = random.randint(1, max_depth)
        ind["alpha"] = 10 ** random.uniform(np.log10(min_alpha), np.log10(max_alpha))
        ind["n_estimators"] = random.randint(min_estimators, max_estimators)
        ind["hidden_size"] = random.randint(min_hidden, max_hidden)

    # mutation of numerical parameters
    if random.random() < param_mut_prob:
        # degree (for poly)
        if random.random() < 0.5:
            ind["degree"] += random.choice([-1, 1])
            ind["degree"] = max(2, min(max_poly_degree, ind["degree"]))

        # n_knots (for spline)
        if random.random() < 0.5:
            ind["n_knots"] += random.choice([-1, 1])
            ind["n_knots"] = max(min_knots, min(max_knots, ind["n_knots"]))

        # max_depth (for tree / rf)
        if random.random() < 0.5:
            ind["max_depth"] += random.choice([-1, 1])
            ind["max_depth"] = max(1, min(max_depth, ind["max_depth"]))

        # alpha (ridge / mlp) — multiplicative mutation in log10
        if random.random() < 0.5:
            log_alpha = np.log10(ind["alpha"])
            log_alpha += random.choice([-0.5, 0.5])
            log_alpha = max(np.log10(min_alpha), min(np.log10(max_alpha), log_alpha))
            ind["alpha"] = 10 ** log_alpha

        # n_estimators (rf)
        if random.random() < 0.5:
            ind["n_estimators"] += random.choice([-5, 5])
            ind["n_estimators"] = max(min_estimators, min(max_estimators, ind["n_estimators"]))

        # hidden_size (mlp)
        if random.random() < 0.5:
            ind["hidden_size"] += random.choice([-2, 2])
            ind["hidden_size"] = max(min_hidden, min(max_hidden, ind["hidden_size"]))

    return ind


def crossover(parent1, parent2):
    """
    Simple uniform crossover: for each gene take value
    from one of the parents.
    """
    child = {}
    for key in parent1.keys():
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child


def ga_model_selection_mdl(
        X,
        y,
        pop_size=30,
        generations=20,
        crossover_prob=0.8,
        mutation_prob=0.4,
        max_poly_degree=8,
        min_knots=3,
        max_knots=10,
        max_depth=6,
        num_immigrants=15,
        mdl_type="cmatrix",
        model_types=None
):
    if model_types is None:
        model_types = MODEL_TYPES_ALL

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    population = [
        random_individual(
            max_poly_degree=max_poly_degree,
            min_knots=min_knots,
            max_knots=max_knots,
            max_depth=max_depth,
            model_types=model_types,
        )
        for _ in range(pop_size)
    ]

    def fitness(individual):
        mdl = evaluate_individual(individual, X, y, mdl_type)
        return -mdl  # maximize -MDL

    # --- GLOBAL BEST ---
    fitness_values = [fitness(ind) for ind in population]
    best_idx = int(np.argmax(fitness_values))
    global_best_ind = population[best_idx]
    global_best_fitness = fitness_values[best_idx]

    for gen in range(generations):
        fitness_values = [fitness(ind) for ind in population]

        # update global best
        gen_best_idx = int(np.argmax(fitness_values))
        gen_best_ind = population[gen_best_idx]
        gen_best_fit = fitness_values[gen_best_idx]
        if gen_best_fit > global_best_fitness:
            global_best_fitness = gen_best_fit
            global_best_ind = gen_best_ind

        # selection (tournament)
        new_population = []
        while len(new_population) < pop_size:
            i1, i2 = random.sample(range(pop_size), 2)
            winner = population[i1] if fitness_values[i1] > fitness_values[i2] else population[i2]
            new_population.append(winner)

        # crossover
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = new_population[i]
            p2 = new_population[(i + 1) % pop_size]
            if random.random() < crossover_prob:
                c1 = crossover(p1, p2)
                c2 = crossover(p2, p1)
            else:
                c1, c2 = p1, p2
            offspring.extend([c1, c2])

        # mutation
        population = []
        for ind in offspring[:pop_size]:
            if random.random() < mutation_prob:
                ind = mutate_individual(
                    ind,
                    max_poly_degree=max_poly_degree,
                    min_knots=min_knots,
                    max_knots=max_knots,
                    max_depth=max_depth,
                    model_types=model_types,
                )
            population.append(ind)

        # --- ELITISM: insert global best ---
        population[0] = global_best_ind.copy()

        # immigrants
        for _ in range(num_immigrants):
            idx = random.randrange(pop_size)
            population[idx] = random_individual(
                max_poly_degree=max_poly_degree,
                min_knots=min_knots,
                max_knots=max_knots,
                max_depth=max_depth,
                model_types=model_types,
            )

        best_mdl_so_far = -global_best_fitness
        print(f"Generation {gen + 1}: best_so_far = {global_best_ind}, MDL = {best_mdl_so_far:.3f}")

    return global_best_ind, -global_best_fitness


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


# additional evaluation function
def evaluate_model_stats(individual, X, y):
    """
    Returns R2, MSE, MAE for a given individual (poly/spline/etc.)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    fit_predict = build_model(individual)
    y_pred, params = fit_predict(X, y)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return {
        "R2": r2,
        "MSE": mse,
        "MAE": mae,
        "n_params": len(params)
    }
