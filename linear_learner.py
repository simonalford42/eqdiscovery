"""reduces force law learning to linear regression"""

from tqdm import tqdm
from nbody import *
from magnet import *
from animate import animate
from enumerate_expressions import *
from boids import load_boids
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn import preprocessing
import group_lasso
from locusts import load_locusts


# from nearest_neighbor import FastNN
from force_learner import *
import numpy as np
import random
import utils
import einops


def sparse_regression(X, y, alpha, feature_cost=None, groups=None, mirror=False, group_reg=None, l1_reg=None):
    """short and modular helper function"""
    if groups is not None:
        num_groups = max(groups) + 1

    if mirror:
        '''
        learn the exact same law for each
        '''
        if feature_cost is not None:
            feature_cost2 = np.zeros(num_groups)

        # all features in the same group get summed to a new feature.
        # after regressing, we use the resulting coefficient for each feature.
        X2 = np.zeros((len(X), num_groups))

        for g in range(num_groups):
            group_indices = np.where(groups == g)[0]

            if feature_cost is not None:
                costs = feature_cost[group_indices]
                assert np.all(costs == costs[0])
                feature_cost2[g] = costs[0]

            X2[:,g] = np.sum(X[:,group_indices], -1)

        old_X = X
        X = X2
        if feature_cost is not None:
            feature_cost = feature_cost2

    scaler = preprocessing.StandardScaler(with_mean=False).fit(X)

    if feature_cost is not None:
        # we expect array of costs for each feature
        # bigger cost is accomplished by making the feature values smaller during the lasso
        # this forces bigger coefficients to use those features
        scaler.scale_ *= np.array(feature_cost)

    X = scaler.transform(X)
    y_scale = np.mean(y*y)**0.5
    y = y/y_scale

    if groups is not None:
        if l1_reg is None:
            l1_reg = alpha
        if group_reg is None:
            group_reg = alpha
        model = group_lasso.GroupLasso(l1_reg=l1_reg, groups=groups, fit_intercept=False, group_reg=group_reg)
    if alpha > 0:
        model = Lasso(fit_intercept=False, alpha=alpha, max_iter=100000)
    else:
        model = LinearRegression(fit_intercept=False)

    model.fit(X, y)
    coefficients = model.coef_

    print(" ==  == finished sparse regression ==  == ")
    print("rank", LinearRegression(fit_intercept=False).fit(X, y).rank_, "/", min(X.shape))
    print("score", model.score(X, y))
    print()

    coefficients = y_scale * coefficients / scaler.scale_

    if mirror:
        # convert back to original basis
        unmirrored_coeffs = np.zeros(old_X.shape[1])
        for g in range(num_groups):
            group_indices = np.where(groups == g)[0]
            unmirrored_coeffs[group_indices] = coefficients[g]
        coefficients = unmirrored_coeffs

    return coefficients


def arrays_proportional(x, y, tolerance=0.02):
    """check if x=ky, but be robust to divide by zero"""
    assert x.shape == y.shape

    k = x/y
    try:
        k = np.mean(k[np.isfinite(k)])
    except TypeError as e:
        print(k)
        import pdb
        pdb.set_trace()
        raise e
    return np.max(np.abs(k*y-x)) < tolerance*np.max(np.abs(x))


def normalize_array(x):
    x = x / np.max(np.abs(x))

    def round_to_sigfigs(x, sigfigs=2):
        p = sigfigs
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    return round_to_sigfigs(x, sigfigs=2)


class AccelerationLearner():
    def __init__(self, dimension, alpha, penalty, basis, cutoff, ops, x, v, cost_dict=None, library_ops=None):
        self.alpha = alpha
        self.dimension = dimension
        self.penalty = penalty
        self.cutoff = cutoff
        self.library_ops = library_ops
        self.ops = ops

        if self.library_ops:
            print('Learned ops: ', [op.expr.pretty_print() for op in self.library_ops])

        if isinstance(basis, int):
            """
            self.basis is a dictionary containing the expressions that serve as basis functions
            self.basis[(n_particles, n_indices)] is an list of expressions for `n_particles` interacting (probably 1 or 2) and n_indices is the number of indices in the return value of the basis function. For example, if the basis function returns a vector, n_indices=1. For a matrix, n_indices=2b
            """
            inputs1 = self.random_inputs(x, v, n_particles=1)
            inputs2 = self.random_inputs(x, v, n_particles=2)

            self.basis = {}
            # construct basis functions both for interaction forces and individual particle forces
            self.basis[(2,1)], self.basis[(2,2)] = construct_basis([], ["R", "V1", "V2"], basis,
                                                                   self.get_ops(["R", "V1", "V2"]),
                                                                   dimension=dimension, cost_dict=cost_dict,
                                                                   inputs=inputs2)

            # remove any interaction forces which do not involve both particles
            self.basis[(2,1)] = [e
                                 for e in self.basis[(2,1)]
                                 if "R" in e.pretty_print() or \
                                 ("V1" in e.pretty_print() and 'V2' in e.pretty_print())]

            self.basis[(2,2)] = [e
                                 for e in self.basis[(2,2)]
                                 if "R" in e.pretty_print() or \
                                 ("V1" in e.pretty_print() and 'V2' in e.pretty_print())]

            self.basis[(1,1)], self.basis[(1,2)]  = construct_basis([], ["V"], basis,
                                                                    self.get_ops(["V"]),
                                                                    dimension=dimension, cost_dict=cost_dict,
                                                                    inputs=inputs1)

            # self.show_basis_function_counts()
        else:
            assert isinstance(basis, dict)
            self.basis=basis

    def get_ops(self, vectors):

        ops = self.ops.copy()

        if self.dimension != 3:
            if Skew in ops:
                ops.remove(Skew)
            if Cross in ops:
                ops.remove(Cross)

        # include library ops if their vector is in the list of vectors
        def all_good_vectors(expr):
            if type(expr) == Vector:
                return expr.name in vectors
            else:
                return all([all_good_vectors(a) for a in expr.arguments()])

        ops += [op for op in self.library_ops if all_good_vectors(op.expr)]
        return ops

    @property
    def basis_size(self):
        return sum(len(functions) for functions in self.basis.values())

    def restrict_basis(self, *choices):
        self.basis = {k: [b for b in bs if b.pretty_print() in choices ]
                      for k, bs in self.basis.items() }

    def show_basis_function_counts(self):
        for i in [1,2]:
            for j in [1,2]:
                print(f"For {i} particles interacting, {j} basis indices:  {len(self.basis[(i,j)])} functions")
                # for b in self.basis[(i,j)]:
                    # print('\t' + b.pretty_print())

    def print_model(self, model, split=False, mirror=False):
        if mirror:
            exprs = set()
            model2 = []
            for (w, (b, t, *o)) in model:
                if b not in exprs:
                    model2.append((w, (b, t, *o)))
                    exprs.add(b)
            model = model2

        if split:
            for w, (basis_expression, t, *object_indices) in sorted(model, key=lambda xx: -abs(xx[0])):
                print(w, "\t", t, "\t", basis_expression.pretty_print(), "\t", *object_indices)
        else:
            for w, (basis_expression, *object_indices) in sorted(model, key=lambda xx: -abs(xx[0])):
                print(w, "\t", basis_expression.pretty_print(), "\t", *object_indices)

    def show_laws(self, split=False, mirror=False):
        laws = self.acceleration_laws
        if mirror:
            exprs = set()
            laws2 = []
            for terms in laws:
                terms2 = []
                for (b, *o) in terms:
                    if b not in exprs:
                        terms2.append((b, *o))
                        exprs.add(b)
                laws2.append(terms2)
            laws = laws2

        print()
        for i, terms in enumerate(laws):
            pretty_terms = []
            for term in terms:
                if split:
                    basis_function, t, i, j, coefficient = term
                else:
                    basis_function, i, j, coefficient = term

                pretty = basis_function.pretty_print()
                if i == j:
                    pretty = pretty.replace("V", f"V_{i}")
                else:
                    pretty = pretty.replace("V1", f"V_{i}").replace("V2", f"V_{j}").replace("R", f"R_{i}{j}")
                if basis_function.return_type == "vector":
                    pretty = "%.03f * %s"%(coefficient, pretty)
                elif basis_function.return_type == "matrix":
                    pretty = "%s [%.03f,%.03f,%.03f]"%(pretty,
                                                       coefficient[0],coefficient[1],coefficient[2])
                else:
                    assert False
                pretty_terms.append(pretty)

            start = f"a_{i}" if not split else f"a_{i}_{t}"
            if pretty_terms:
                print(start + f" = {' + '.join(pretty_terms)}")

    def check_goal_exprs_present_in_basis(self):
        matches = 0
        for basis_fns in self.basis.values():
            for b in basis_fns:
                if b.pretty_print() in GOAL_EXPRS:
                    matches += 1
                    print('Goal expression still in basis:', b.pretty_print())
        if matches <= 1:
            print('Goal expression(s) missing from basis')


    def remove_from_basis(self, fns_to_remove):
        self.valuation_dictionary = {(b, *rest): value
                                for (b, *rest), value in self.valuation_dictionary.items()
                                if b not in fns_to_remove}
        self.basis = {b_key: [b for b in b_value if b not in fns_to_remove]
                      for b_key, b_value in self.basis.items()}
        # self.check_goal_exprs_present_in_basis()
        print('New basis size: ', self.basis_size)


    def random_inputs(self, x, v, n_particles, n=10):
        t_s = random.choices(range(x.shape[0]), k=n)
        i_s = random.choices(range(x.shape[1]), k=n)
        if n_particles == 1:
            j_s = i_s
        else:
            j_s = random.choices(range(x.shape[1]), k=n)

        inputs = []
        for t, i, j in zip(t_s, i_s, j_s):
            if i == j:
                input_dictionary = {"V":v[t,i]}
            else:
                input_dictionary = {"R": x[t,j]-x[t,i],
                                    "V1": v[t,i],
                                    "V2": v[t,j]}
            inputs.append(input_dictionary)

        return inputs


    def evaluate_basis_functions(self, x, v):
        """
        returns: dictionary mapping (b, t, i, j) to its vector or matrix valuation, where:
        t index is time
        i index is particle that force is acting on
        j index is particle generating the force
        when i=j, this is not an interaction force
        """
        self.valuation_dictionary = {}
        T, N, D = x.shape
        assert D == self.dimension

        problematic_functions = set() # these are features that give nan/infs, or are otherwise invalid
        # self.check_goal_exprs_present_in_basis()

        if N == 1:
            problematic_functions = {b
                                     for (n_particles, n_indices), bs in self.basis.items()
                                     if n_particles == 2
                                     for b in bs}

        valuation_dictionary2 = {}

        for t in tqdm(range(T)):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        n_particles = 1
                        input_dictionary = {"V":v[t,i]}
                    else:
                        n_particles = 2
                        input_dictionary = {"R": x[t,j]-x[t,i],
                                            "V1": v[t,i],
                                            "V2": v[t,j]}

                    for n_indices in [1,2]:
                        for b in self.basis[(n_particles, n_indices)]:
                            if b in problematic_functions: continue

                            value = b.evaluate(input_dictionary)

                            if "nan" in str(value) or "inf" in str(value):
                                problematic_functions.add(b)
                            else:
                                self.valuation_dictionary[(b,t,i,j)] = value
                                if b in valuation_dictionary2:
                                    valuation_dictionary2[b].append(value)
                                else:
                                    valuation_dictionary2[b] = [value]

        all_zero_functions = set()
        too_small_functions = set()
        for b in valuation_dictionary2:
            max_val = max([abs(v).max() for v in valuation_dictionary2[b]])

            if max_val == 0:
                all_zero_functions.add(b)
            elif max_val <= 1E-6:
                too_small_functions.add(b)

        for fns, cause in zip(
            [problematic_functions, too_small_functions, all_zero_functions],
            ["cause numerical instability", "are too small", "are all zeros"]):
            print("removing ", len(fns), "/", self.basis_size,
                  "basis functions that", cause)

        bad_functions = problematic_functions.union(too_small_functions).union(all_zero_functions)
        # bad_functions = problematic_functions  # disable removing the other fns
        self.remove_from_basis(bad_functions)

        # Now let's figure out if any of the basis functions are just linear rescalings of others
        # Compute the signature of each basis function, which is the vector of its valuations
        # Check if any signatures are in constant proportion
        problematic_functions = set()
        signature = {}
        for (n_particles, n_indices), bs in self.basis.items():
            for b in bs:
                if n_particles == 1:
                    particle_pairs = [(i,i) for i in range(N)]
                else:
                    particle_pairs = [(i,j) for i in range(N) for j in range(N) if i != j]

                sig = np.stack([self.valuation_dictionary[(b, t, i, j)]
                                for t in range(T)
                                for i,j in particle_pairs])
                sig = np.reshape(sig, -1)
                sig = sig / np.linalg.norm(sig)

                signature[b] = sig

        for n_particles in [1,2]:
            for n_indices in [1,2]:
                for n, b1 in enumerate(self.basis[(n_particles, n_indices)]):
                    for b2 in self.basis[(n_particles, n_indices)][:n]:
                        if arrays_proportional(signature[b1], signature[b2]):
                            problematic_functions.add(b1)
                            break

        print("Removing ", len(problematic_functions), "/", self.basis_size,
              "basis functions that are redundant")
        self.remove_from_basis(problematic_functions)

        return self.valuation_dictionary


    def fit(self, x, v, a):

        # extract shapes and make sure everything has the right dimensions
        T = x.shape[0]
        N = x.shape[1]
        if self.dimension != x.shape[2]:
            missing_dimensions = self.dimension - x.shape[2]
            assert missing_dimensions > 0
            # expand to give extra dimension
            x = np.concatenate([x, np.zeros((*x.shape[:-1], missing_dimensions))], -1)
            v = np.concatenate([v, np.zeros((*v.shape[:-1], missing_dimensions))], -1)
            a = np.concatenate([a, np.zeros((*a.shape[:-1], missing_dimensions))], -1)

        D = x.shape[2]
        assert D == self.dimension

        # see comment for structure of valuations
        valuations = self.evaluate_basis_functions(x, v)
        self.valuations = valuations

        # Construct the regression problem
        # We are predicting acceleration
        Y = []
        X = []

        # make group ids for the different basis functions
        # we put all of the basis terms of the same eq but different particle targets into a group.
        # basis[(i, j)], i is the number of particles, j is the number of indices in the return value
        basis_ids = {}
        current_id = 0
        for n_particles in [1,2]:
            for b in self.basis[(n_particles, 1)]:
                if b not in basis_ids:
                    basis_ids[b] = current_id
                    current_id += 1

            for b in self.basis[(n_particles, 2)]:
                for u in range(D):
                    if (b, u) not in basis_ids:
                        basis_ids[(b, u)] = current_id
                        current_id += 1

        for t in range(T):
            for i in range(N):
                for d in range(D):
                    Y.append(a[t,i,d])
                    feature_dictionary = {}

                    for j in range(N):
                        n_particles = 1 if i == j else 2

                        # valuation is a vector (1-indexed)
                        # interaction multiplier is a scaler
                        for b in self.basis[(n_particles,1)]:
                            feature_dictionary[(b,i,j)] = valuations[(b,t,i,j)][d]

                        # valuation is a matrix (2-indexed)
                        # interaction multiplier is a vector
                        for b in self.basis[(n_particles,2)]:
                            for u in range(D):
                                feature_dictionary[(b,i,j,u)] = valuations[(b,t,i,j)][d,u]

                    X.append(feature_dictionary)

        feature_names = list(sorted({ fn for fd in X for fn in fd.keys() }))

        if arguments.group or arguments.mirror:
            groups = np.array([basis_ids[(b,u) if len(rest) == 2 else b] for b, *rest, u in feature_names])
        else:
            groups = None

        X_matrix = np.array([ [ fd.get(f, 0.) for f in feature_names ] for fd in X ])
        Y = np.array(Y)

        if arguments.split:
            '''
            Split up basis functions into different time segments (zero padding outside that segment)
            and have a group for each original basis function consisting of the different time segments.
            then run group lasso. this way, the model tries to minimize the number of total basis functions,
            but is allowed to do whatever it wants when fitting a given time segment.
            '''
            F = X_matrix.shape[1]
            L = arguments.split_length
            assert T % L == 0, f"T must be divisible by L, but T={T}, and L={L}"
            T1 = int(T / L)
            X_matrix = einops.repeat(X_matrix, 'TND F -> TND F R', R=T1)
            X_matrix = einops.rearrange(X_matrix, '(T ND) F R -> T ND F R', T=T)
            # if array of length T is split up into T1 chunks of length L,
            # then the t'th chunk has indices [t*L, (t+1)*L)
            # for feature (f, t), everything outside [t*L, (t+1)*L) should be zero
            for t in range(T1):
                # everything outside [t*L, (t+1)*L) should be zero
                X_matrix[:t*L, :, :, t] = 0
                X_matrix[(t+1)*L:, :, :, t] = 0

            X_matrix = einops.rearrange(X_matrix, 'T ND F R -> (T ND) (F R)')
            # F groups, each of size T1
            groups = einops.repeat(np.arange(F), 'F -> (F T1)', T1=T1)
            assert_equal(len(groups), X_matrix.shape[1])
            feature_names = [(f, t, *rest) for (f, *rest) in feature_names for t in range(T1)]

        feature_cost = np.array([ self.penalty*int(basis_function.return_type == "matrix") + 1
                                for (basis_function, *rest) in feature_names ])

        coefficients = sparse_regression(X_matrix, Y, alpha=arguments.alpha, feature_cost=feature_cost, groups=groups, mirror=arguments.mirror)

        model = [(coefficients[feature_index], feature_name)
                 for feature_index, feature_name in enumerate(feature_names)
                 if abs(coefficients[feature_index]) > self.cutoff
        ]

        self.print_model(model, split=arguments.split, mirror=arguments.mirror)

        # did we just shrink the basis?
        surviving_functions = {basis_function for w, (basis_function, *objects) in model }
        original_functions = {basis_function for (basis_function, *objects) in feature_names }
        if len(surviving_functions) < len(original_functions):
            print(f"Basis shrunk. Reestimating with smaller basis of {len(surviving_functions)} functions")
            new_basis = {b_key: [b for b in b_value if b in surviving_functions ]
                         for b_key, b_value in self.basis.items()}
            return AccelerationLearner(self.dimension, self.alpha, self.penalty, new_basis, self.cutoff, self.ops, x, v).fit(x, v, a)
        else:
            print(" ==  == acceleration learning has converged, reestimating coefficients ==  == ")

            coefficients = sparse_regression(X_matrix, Y, alpha=0, groups=groups, mirror=arguments.mirror)

            model = [(coefficients[feature_index], feature_name)
                     for feature_index, feature_name in enumerate(feature_names)
                     if abs(coefficients[feature_index]) > 1e-3]

            self.print_model(model, split=arguments.split, mirror=arguments.mirror)

            # now we convert to acceleration laws
            self.acceleration_laws = []
            for i in range(N):
                for t in range(1 if not arguments.split else T1):
                    this_law = []
                    for n_particles in [1,2]:
                        for n_indices in [1,2]:
                            for b in self.basis[(n_particles, n_indices)]:

                                if n_particles == 1: others = [i]
                                else: others = [o for o in range(N) if o != i ]

                                for j in others:
                                    if arguments.split:
                                        matches = [ (w, t1, *more_objects)
                                                    for w, (basis_expression, t1, object1, *more_objects) in model
                                                    if basis_expression == b and t1 == t and object1 == i and more_objects[0] == j]
                                    else:
                                        matches = [ (w, *more_objects)
                                                    for w, (basis_expression, object1, *more_objects) in model
                                                    if basis_expression == b and object1 == i and more_objects[0] == j]

                                    if len(matches) == 0: continue

                                    if n_indices == 1:
                                        assert len(matches) == 1
                                        if arguments.split:
                                            this_law.append((b, i, j, t, matches[0][0]))
                                        else:
                                            this_law.append((b, i, j, matches[0][0]))

                                    if n_indices == 2:
                                        assert len(matches) <= 3
                                        latent_vector = np.zeros(3)
                                        if arguments.split:
                                            for coefficient, t, _, idx in matches: latent_vector[idx] = coefficient
                                            this_law.append((b, i, j, t, latent_vector))
                                        else:
                                            for coefficient, _, idx in matches: latent_vector[idx] = coefficient
                                            this_law.append((b, i, j, latent_vector))

                    self.acceleration_laws.append(this_law)

            self.show_laws(split=arguments.split, mirror=arguments.mirror)

            return self


def run_linear_learner(arguments, data_dict):
    iters = 2 if arguments.basis2 > 0 else 1
    def basis_size(iter):
        if iter == 0: return arguments.basis
        if iter == 1: return arguments.basis2
        assert False

    is_experiment_solved = {name: False for name in data_dict}
    library_ops = []
    # library_ops = [abstraction(pretty_print_to_expr('(v/ (hat R) (dp R R))'))]

    for iteration in range(iters):
        expressions = []

        for name in data_dict:
            if is_experiment_solved[name]:
                continue
            print(f'Testing physics learner on {name}')

            x, v, _, a = data_dict[name]
            dimension = 3 if arguments.embed else x.shape[-1]

            al = AccelerationLearner(dimension,
                                     arguments.alpha,
                                     arguments.penalty,
                                     basis_size(iteration),
                                     arguments.cutoff,
                                     ops=arguments.ops,
                                     library_ops=library_ops,
                                     x=x, v=v)

            al = al.fit(x, v, a)
            print()
            exprs = []
            num_terms = 0
            for law in al.acceleration_laws:
                for term in law:
                    if arguments.split:
                        (expr, t, i, j, c) = term
                    else:
                        (expr, i, j, c) = term
                    exprs.append(expr)
                    num_terms += sum(c != 0) if type(c) == np.ndarray else 1

            n = x.shape[1]
            if num_terms < 5 * n:
                print(f'Solved {name}')
                is_experiment_solved[name] = True
            else:
                print(f'Did not solve {name} (uses {num_terms} > {5*n} terms)')

            expressions += exprs

            if arguments.force:
                fl = ForceLearner(0, arguments.lines)
                fl.fit(al.acceleration_laws)

            if arguments.animate_learned and iteration == iters-1:
                # animate using the learned laws
                x = simulate_learned_laws(x[0], v[0], al.acceleration_laws)
                animate(x[::x.shape[0]//100], fn=name)


        if iters > 1 and iteration == 0:
            # library learning with the returned expressions
            pretty_prints = library_learn(expressions)
            library_ops = [abstraction(pretty_print_to_expr(s)) for s in pretty_prints]




OPSET_DICT = {
    'default': [
        Divide,
        Outer,
        ScaleInverse,
        Hat,
        Length,
        Scale,
        Inner,
        Times,
        Cross,
        Skew,
    ],
    'boids': [
        Within35,
        # Within100,
        Length,
        Scale,
    ],
}


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--simulation", "-s", default='magnet2')
    parser.add_argument("--alpha", "-a", default=1e-3, type=float, help="controls sparsity")
    parser.add_argument("--embed", "-e", default=False, action="store_true", help="embed in 3d")
    parser.add_argument("--penalty", "-p", default=1.5, type=float,
                        help="penalty for introducing latent vectors")
    parser.add_argument("--animate", "-m", default=False, action="store_true")
    parser.add_argument("--basis", "-b", default=200, type=int, help="number of basis functions")
    parser.add_argument("--basis2", "-b2", default=0, type=int, help="number of basis functions for second round. by default, will not do a second round")
    parser.add_argument("--latent", "-l", default=0, type=int, help="number of latent parameters to associate with each particle (in addition to its mass) ")
    parser.add_argument("--lines", "-L", default=3, type=int, help="number of lines of code to synthesize per coefficient")
    parser.add_argument("--cutoff", "-c", default=1E-4, type=float, help="remove functions with coefficients below this value during sparse regression")
    parser.add_argument("--force", '-f', action="store_true", help="run force learning")
    parser.add_argument("--noise", '-n', action="store_true", help="add noise to the data")
    parser.add_argument("--noise_intensity", '-ni', default=0.01 , type=float, help="std of noise to add to the data")
    parser.add_argument("--animate_learned", '-al', action='store_true', help='animate the learned acceleration laws')
    parser.add_argument("--group", '-g', action='store_true', help='run group lasso so that learned terms are same for different particle pairs')
    parser.add_argument("--mirror", '-mi', action='store_true', help='learn identical laws for each particle')
    parser.add_argument("--split", '-sp', action='store_true', help='split the data into chunks of length split_length')
    parser.add_argument("--split_length", '-sl', default=1, type=int, help='length to split sequence into, if --split is enabled')
    parser.add_argument("--opset", '-o', default=None, type=str, help='op set to use')
    parser.add_argument("--start", default=0, type=int, help='start ix for locust data')

    arguments = parser.parse_args()
    assert not (arguments.split and arguments.group), 'split and group are incompatible'

    key = arguments.simulation
    if arguments.opset:
        key = arguments.opset
    elif arguments.simulation in OPSET_DICT:
        key = arguments.simulation
    else:
        key = 'default'

    arguments.ops = OPSET_DICT[key]

    experiments = [
        ("drag3", simulate_drag3),
        ("falling", simulate_falling),
        ("orbit", simulate_circular_orbit),
        ("orbit2", simulate_2_orbits),
        ("drag1", simulate_drag1),
        ("drag2", simulate_drag2),
        ("magnet1", simulate_charge_in_uniform_magnetic_field),
        ("magnet2", simulate_charge_dipole),
        ("boids", lambda: load_boids(i=3)),
        ("spring", simulate_elastic_pendulum),
        ("locusts", lambda: load_locusts('01EQ20191203_tracked.csv', T=1000, speedup=1, start=arguments.start)),
        # ("locusts", lambda: load_locusts('01EQ20191203_tracked.csv', T=4000, speedup=1, start=3000)),
        ("circle", simulate_circle),
    ]

    if arguments.simulation != 'all':
        experiments = [(name, callback) for name, callback in experiments
                                        if name == arguments.simulation]
        if len(experiments) == 0:
            raise ValueError(f'Unknown simulation: {arguments.simulation}')
    else:
        experiments = [e for e in experiments if e[0] not in ['spring', 'boids']]

    print(f'{arguments=}')
    data_dict = {}
    for name, callback in experiments:
        data_dict[name] = callback()

        if arguments.noise:
            data_dict[name] = utils.noisify(data_dict[name], intensity=arguments.noise_intensity)

        print(f"simulated {name} data")

        if arguments.animate:
            x, _, _, _ = data_dict[name]
            # animate(x[::x.shape[0]//100], fn=name)
            # animate_spring(x[::x.shape[0]//100])
            # print(f"animated spring")
            # import sys; sys.exit(0)

    run_linear_learner(arguments, data_dict)

