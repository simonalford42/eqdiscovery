"""reduces force law learning to linear regression"""

from tqdm import tqdm
from nbody import *
from magnet import *
from animate import animate
from enumerate_expressions import *
# from nearest_neighbor import FastNN
from force_learner import *
import numpy as np
import random

def sparse_regression(X, y, alpha=1e-3, feature_cost=None):
    """short and modular helper function"""
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn import preprocessing

    scaler = preprocessing.StandardScaler(with_mean=False).fit(X)

    if feature_cost:
        # we expect array of costs for each feature
        # bigger cost is accomplished by making the feature values smaller during the lasso
        # this forces bigger coefficients to use those features
        scaler.scale_ *= np.array(feature_cost)

    X = scaler.transform(X)

    y_scale = np.mean(y*y)**0.5
    #print("y_scale", y_scale)
    y = y/y_scale

    if alpha > 0:
        model = Lasso(fit_intercept=False, alpha=alpha, max_iter=100000)
    else:
        model = LinearRegression(fit_intercept=False)

    model.fit(X, y)

    print(" ==  == finished sparse regression ==  == ")
    print("rank", LinearRegression(fit_intercept=False).fit(X, y).rank_, "/", min(X.shape))
    print("score", model.score(X, y))
    print()

    return y_scale*model.coef_ / scaler.scale_


def arrays_proportional(x, y, tolerance=0.02):
    """check if x=ky, but be robust to divide by zero"""
    assert x.shape == y.shape

    k = x/y
    k = np.mean(k[np.isfinite(k)])
    return np.max(np.abs(k*y-x)) < tolerance*np.max(np.abs(x))


class AccelerationLearner():
    def __init__(self, dimension, alpha, penalty, basis, weighted=False):
        self.alpha = alpha
        self.dimension = dimension
        self.penalty = penalty

        if isinstance(basis, int):
            """
            self.basis is a dictionary containing the expressions that serve as basis functions
            self.basis[(n_particles, n_indices)] is an list of expressions for `n_particles` interacting (probably 1 or 2) and n_indices is the number of indices in the return value of the basis function. For example, if the basis function returns a vector, n_indices=1. For a matrix, n_indices=2b
            """
            self.basis = {}
            # construct basis functions both for interaction forces and individual particle forces
            self.basis[(2,1)], self.basis[(2,2)] = construct_basis([], ["R", "V1", "V2"], basis,
                                                                   dimension=dimension, weighted=weighted)

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
                                                                    dimension=dimension, weighted=weighted)

            self.show_basis_function_counts()
        else:
            assert isinstance(basis, dict)
            self.basis=basis


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

    def show_laws(self):
        print()
        for i, terms in enumerate(self.acceleration_laws):

            pretty_terms = []
            for basis_function, i, j, coefficient in terms:
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

            print(f"a_{i} = {' + '.join(pretty_terms)}")

    def evaluate_basis_functions(self, x, v):
        """
        returns: dictionary mapping (b, t, i, j) to its vector or matrix valuation, where:
        t index is time
        i index is particle that force is acting on
        j index is particle generating the force
        when i=j, this is not an interaction force
        """

        valuation_dictionary = {}
        T = x.shape[0]
        N = x.shape[1]
        D = x.shape[2]
        assert D == self.dimension


        problematic_functions = set() # these are features that give nan/infs, or are otherwise invalid
        if N == 1:
            problematic_functions = {b
                                     for (n_particles, n_indices), bs in self.basis.items()
                                     if n_particles == 2
                                     for b in bs}

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
                                valuation_dictionary[(b,t,i,j)] = value

        print("Removing ", len(problematic_functions), "/", self.basis_size,
              "basis functions that cause numerical instability")
        #print({pf.pretty_print() for pf in problematic_functions })

        valuation_dictionary = {(b, *rest): value
                                for (b, *rest), value in valuation_dictionary.items()
                                if b not in problematic_functions}

        self.basis = {b_key: [b for b in b_value if b not in problematic_functions ]
                      for b_key, b_value in self.basis.items()}

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

                sig = np.stack([valuation_dictionary[(b, t, i, j)]
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
                            print(b1.pretty_print(), "made redundant by", b2.pretty_print())
                            break

        print("Removing ", len(problematic_functions), "/", self.basis_size,
              "basis functions that are redundant:")
        #print({pf.pretty_print() for pf in problematic_functions })


        valuation_dictionary = {(b, *rest): value
                                for (b, *rest), value in valuation_dictionary.items()
                                if b not in problematic_functions}
        self.basis = {b_key: [b for b in b_value if b not in problematic_functions ]
                      for b_key, b_value in self.basis.items()}

        self.show_basis_function_counts()


        return valuation_dictionary


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

        # Construct the regression problem
        # We are predicting acceleration
        Y = []
        X = []

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

        X_matrix = np.array([ [ fd.get(f, 0.) for f in feature_names ] for fd in X ])
        Y = np.array(Y)

        feature_cost = [ self.penalty*int(basis_function.return_type == "matrix") + 1
                         for (basis_function, *rest) in feature_names ]

        coefficients = sparse_regression(X_matrix, Y, alpha=self.alpha, feature_cost=feature_cost)

        model = [(coefficients[feature_index], feature_name)
                 for feature_index, feature_name in enumerate(feature_names)
                 if abs(coefficients[feature_index]) > 1e-4
        ]

        for w, (basis_expression, *object_indices) in sorted(model, key=lambda xx: -abs(xx[0])):
            print(w, "\t", basis_expression.pretty_print(), "\t", *object_indices)

        # did we just shrink the basis?
        surviving_functions = {basis_function for w, (basis_function, *objects) in model }
        original_functions = {basis_function for (basis_function, *objects) in feature_names }
        if len(surviving_functions) < len(original_functions):
            print(f"Basis shrunk. Reestimating with smaller basis of {len(surviving_functions)} functions")
            new_basis = {b_key: [b for b in b_value if b in surviving_functions ]
                         for b_key, b_value in self.basis.items()}
            return AccelerationLearner(self.dimension, self.alpha, self.penalty, new_basis).fit(x, v, a)
        else:
            print(" ==  == acceleration learning has converged, reestimating coefficients ==  == ")

            coefficients = sparse_regression(X_matrix, Y, alpha=0)
            model = [(coefficients[feature_index], feature_name)
                     for feature_index, feature_name in enumerate(feature_names)
                     if abs(coefficients[feature_index]) > 1e-3]

            for w, (basis_expression, *object_indices) in sorted(model, key=lambda xx: -abs(xx[0])):
                print(w, "\t", basis_expression.pretty_print(), "\t", *object_indices)

            # now we convert to acceleration laws
            self.acceleration_laws = []
            for i in range(N):
                this_law = []
                for n_particles in [1,2]:
                    for n_indices in [1,2]:
                        for b in self.basis[(n_particles, n_indices)]:

                            if n_particles == 1: others = [i]
                            else: others = [o for o in range(N) if o != i ]

                            for j in others:
                                matches = [ (w, *more_objects)
                                            for w, (basis_expression, object1, *more_objects) in model
                                            if basis_expression == b and object1 == i and more_objects[0] == j]
                                if len(matches) == 0: continue

                                if n_indices == 1:
                                    assert len(matches) == 1
                                    this_law.append((b, i, j, matches[0][0]))

                                if n_indices == 2:
                                    assert len(matches) <= 3
                                    latent_vector = np.zeros(3)
                                    for coefficient, _, idx in matches: latent_vector[idx] = coefficient
                                    this_law.append((b, i, j, latent_vector))

                self.acceleration_laws.append(this_law)

            self.show_laws()

            return self

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--simulation", "-s")
    parser.add_argument("--alpha", "-a", default=1e-3, type=float, help="controls sparsity")
    parser.add_argument("--embed", "-e", default=False, action="store_true", help="embed in 3d")
    parser.add_argument("--penalty", "-p", default=10, type=float,
                        help="penalty for introducing latent vectors")
    parser.add_argument("--animate", "-m", default=False, action="store_true")
    parser.add_argument("--basis", "-b", default=200, type=int, help="number of basis functions")
    parser.add_argument("--latent", "-l", default=0, type=int, help="number of latent parameters to associate with each particle (in addition to its mass) ")
    parser.add_argument("--lines", "-L", default=3, type=int, help="number of lines of code to synthesize per coefficient")
    parser.add_argument("--weighted", "-P", default=False, action="store_true", help="use weighted enumeration for basis functions")
    arguments = parser.parse_args()

    for name, callback in [
            ("drag3", simulate_drag3),
            ("falling", simulate_falling),
            ("orbit", simulate_circular_orbit),
            ("orbit2", simulate_2_orbits),
            ("drag1", simulate_drag1),
            ("drag2", simulate_drag2),
            ("magnet1", simulate_charge_in_uniform_magnetic_field),
            ("magnet2", simulate_charge_dipole),

    ]:
        if arguments.simulation and arguments.simulation != name:
            continue
        print(f"Testing physics learner on {name}")
        x, v, f, a = callback()

        print(f"simulated {name} data")

        if arguments.animate:
            animate(x[::x.shape[0]//100], fn=name)
            print(f"animated physics into {name}")

        dimension = 3 if arguments.embed else x.shape[-1]

        al = AccelerationLearner(dimension,
                                 arguments.alpha,
                                 arguments.penalty,
                                 arguments.basis,
                                 arguments.weighted)
        al = al.fit(x, v, a)

        # fl = ForceLearner(0, arguments.lines)
        # fl.fit(al.acceleration_laws)


        print()
        print()
        print()
        print()
        print()
        print()
