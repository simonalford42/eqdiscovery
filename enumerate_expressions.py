import itertools
import random
import numpy as np
import math


class Expression():

    def evaluate(self, environment):
        assert False, "not implemented"

    def arguments(self):
        assert False, "not implemented"

    def cost(self):
        return 1 + sum([0] + [argument.cost() for argument in self.arguments()])

    @property
    def return_type(self):
        return self.__class__.return_type

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self): return hash(str(self))

    def __ne__(self, other): return str(self) != str(other)

    def __gt__(self, other): return str(self) > str(other)

    def __lt__(self, other): return str(self) < str(other)


class Real(Expression):
    return_type = "real"
    argument_types = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Real('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        return environment[self.name]

    def arguments(self): return []

class Vector(Expression):
    return_type = "vector"
    argument_types = []

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Vector('{self.name}')"

    def pretty_print(self):
        return self.name

    def evaluate(self, environment):
        return environment[self.name]

    def arguments(self): return []

class Plus(Expression):
    return_type = "real"
    argument_types = ["real","real"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Plus({self.x}, {self.y})"

    def pretty_print(self):
        return f"(+ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        return x + y

    def arguments(self): return [self.x, self.y]

class Times(Expression):
    return_type = "real"
    argument_types = ["real","real"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Times({self.x}, {self.y})"

    def pretty_print(self):
        return f"(* {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        return x * y

    def arguments(self): return [self.x, self.y]

class Divide(Expression):
    return_type = "real"
    argument_types = ["real","real"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Divide({self.x}, {self.y})"

    def pretty_print(self):
        return f"(/ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        #y[np.] = np.clip(y, 1e-3, None)
        return  x / y

    def arguments(self): return [self.x, self.y]


class Reciprocal(Expression):
    return_type = "real"
    argument_types = ["real"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Reciprocal({self.x})"

    def pretty_print(self):
        return f"(1/ {self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        return 1/x

    def arguments(self): return [self.x, self.y]

class Inner(Expression):
    return_type = "real"
    argument_types = ["vector","vector"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Inner({self.x}, {self.y})"

    def pretty_print(self):
        return f"(dp {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        if isinstance(x, np.ndarray):
            return np.sum(x * y, -1)
        return (x * y).sum(-1).unsqueeze(-1)

    def arguments(self): return [self.x, self.y]

class Cross(Expression):
    return_type = "vector"
    argument_types = ["vector","vector"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Cross({self.x}, {self.y})"

    def pretty_print(self):
        return f"(X {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        if isinstance(x, np.ndarray):
            return np.cross(x, y)
        assert False
        return (x * y).sum(-1).unsqueeze(-1)

    def arguments(self): return [self.x, self.y]

class Outer(Expression):
    return_type = "matrix"
    argument_types = ["vector","vector"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Outer({self.x}, {self.y})"

    def pretty_print(self):
        return f"(op {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        if isinstance(x, np.ndarray):
            return np.outer(x, y)
        assert False
        return (x * y).sum(-1).unsqueeze(-1)

    def arguments(self): return [self.x, self.y]

class Skew(Expression):
    return_type = "matrix"
    argument_types = ["vector"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Skew({self.x})"

    def pretty_print(self):
        return f"(skew {self.x.pretty_print()})"

    def evaluate(self, environment):
        v = self.x.evaluate(environment)
        if isinstance(v, np.ndarray):
            x, y, z = v[0], v[1], v[2]

            return np.array([[0, -z, y],
                             [z, 0, -x],
                             [-y, x, 0]])
        assert False
        return (x * y).sum(-1).unsqueeze(-1)

    def arguments(self): return [self.x]


class Length(Expression):
    return_type = "real"
    argument_types = ["vector"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Length({self.x})"

    def pretty_print(self):
        return f"(len {self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        return np.sum(x * x)**0.5

    def arguments(self): return [self.x]


class Scale(Expression):
    return_type = "vector"
    argument_types = ["real","vector"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"Scale({self.x}, {self.y})"

    def pretty_print(self):
        return f"(* {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        return x * y

    def arguments(self): return [self.x, self.y]

class ScaleInverse(Expression):
    return_type = "vector"
    argument_types = ["vector", "real"]

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"ScaleInverse({self.x}, {self.y})"

    def pretty_print(self):
        return f"(/ {self.x.pretty_print()} {self.y.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        y = self.y.evaluate(environment)
        return x / y

    def arguments(self): return [self.x, self.y]

class Hat(Expression):
    return_type = "vector"
    argument_types = ["vector"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"Hat({self.x})"

    def pretty_print(self):
        return f"(hat {self.x.pretty_print()})"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if isinstance(x, np.ndarray):
            norm = np.sum(x * x, -1)**0.5
            if norm > 0: return x/norm
            else: return np.zeros(x.shape)
        else:
            return x/(((x*x).sum(-1)**0.5).unsqueeze(-1))

    def arguments(self): return [self.x]


def weighted_bottom_up_generator(cost_bound, operators, constants, inputs, cost_dict=None):
    """
    cost_bound: int. an upper bound on the cost of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    inputs: list of environments (the input), such as [{'x': 5}, {'x': 1}]
    cost_dict: dict mapping Expression subclass to their cost. costs must be integral. if none provided, does uniform cost
    yields: sequence of programs, ordered by expression cost, which are semantically distinct on the input examples
    """

    variables = infer_variables(inputs)
    variables_and_constants = constants + variables

    if cost_dict is None:
        cost_dict = {op: 1 for op in operators}
        cost_dict.update({type(v): 1 for v in variables_and_constants})

    assert all(isinstance(cost, int) for cost in cost_dict.values()), 'costs must be integral'

    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size
    observed_values = set()

    R = Vector('R')
    V1 = Vector('V1')
    goal_expression = Divide(Cross(R, V1), Times(Inner(R, R), Inner(R, R)))
    goal_valuation = np.array([goal_expression.evaluate(input) for input in inputs])
    goal_valuation = goal_valuation / np.linalg.norm(goal_valuation)
    goal_v1 = np.around(goal_valuation*100, decimals=5).tobytes()

    enumerated_expressions = {}
    def record_new_expression(expression, cost):
        """Returns True iff the semantics of this expression has never been seen before"""
        nonlocal inputs, observed_values

        valuation = np.array([expression.evaluate(input) for input in inputs])

        # discard all zeros
        if np.max(np.abs(valuation)) < 1e-5:
            return False # bad expression

        # discard invalid
        if np.any(~np.isfinite(valuation)):
            return False

        # homogeneity assumption:
        # we only care about the direction, not rescaling or sign changes
        valuation = valuation / np.linalg.norm(valuation)

        # if (expression.pretty_print().startswith('(/ (X R V1)')
            # or expression.pretty_print().startswith('(/ (X V1 R)')):
            # print(expression.pretty_print())

        # things we hash
        v1 = np.around(valuation*100, decimals=5).tobytes()
        v2 = np.around(-valuation*100, decimals=5).tobytes()

        if v1 == goal_v1 or v2 == goal_v1:
            print(f'found a match: {expression.pretty_print()} in {len(observed_values)} steps with cost {cost}')
            assert False

        # is this something we have not seen before?
        if v1 not in observed_values and v2 not in observed_values:
            observed_values.add(v1)

            # we have some new behavior
            key = (expression.__class__.return_type, cost)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, (v1, len(observed_values))))
            return True

        return False

    lvl = 1
    while True:
        for terminal in variables_and_constants:
            if cost_dict[type(terminal)] == lvl:
                if record_new_expression(terminal, lvl):
                    yield terminal

        for operator in operators:
            op_cost = cost_dict[operator]
            if op_cost < lvl:
                for arg_costs in integer_partitions(lvl - op_cost, len(operator.argument_types)):
                    candidate_arguments = [enumerated_expressions.get((typ, cost), [])
                                           for typ, cost in zip(operator.argument_types, arg_costs)]

                    for arguments in itertools.product(*candidate_arguments):
                        # note: could make use of precomputed evaluation of
                        # subtrees when evaluating later
                        new_expression = operator(*[e for e,v in arguments])
                        if record_new_expression(new_expression, lvl):
                            if len(observed_values) % 1000 == 0:
                                print(f'{len(observed_values)} expressions')
                            yield new_expression

        lvl += 1
        if cost_bound and lvl > cost_bound:
            break



def integer_partitions(target_value, number_of_arguments):
    """ Returns all ways of summing up to `target_value` by adding
        `number_of_arguments` nonnegative integers.
     """

    if target_value < 0:
        return []

    if number_of_arguments == 1:
        return [[target_value]]

    return [ [x1] + x2s
             for x1 in range(target_value + 1)
             for x2s in integer_partitions(target_value - x1, number_of_arguments - 1) ]


def infer_variables(inputs):
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)

    return list({make_variable(variable_name, variable_value)
                      for input in inputs
                      for variable_name, variable_value in input.items() })


basis_cache = {}
def construct_basis(reals, vectors, size, dimension=3, weighted=False):
    basis_key = (tuple(reals), tuple(vectors), size, dimension, weighted)
    if basis_key in basis_cache: return basis_cache[basis_key]

    operators = [Hat, Outer, Inner, Divide, Times, Scale, Reciprocal, ScaleInverse, Length]
    if dimension == 3: operators.extend([Skew, Cross])

    def random_input():
        d = {}
        for nm in reals:
            d[nm] = random.random()*10
        for nm in vectors:
            d[nm] = np.random.random(dimension)*10-5
        return d

    inputs = [random_input()
              for _ in range(10) ]
    vector_basis = []
    matrix_basis = []

    variables = infer_variables(inputs)
    print(f'{variables=}')
    constants = []
    variables_and_constants = variables + constants

    cost_dict = None
    if weighted:
        cost_dict = {op: 500 for op in operators}
        cost_dict.update({type(v): 1 for v in variables_and_constants})
        for term in [Inner, ScaleInverse, Times, Cross]:
            cost_dict[term] = 1

    for expression in weighted_bottom_up_generator(20, operators, constants,
                                                  inputs, cost_dict=cost_dict):
        if expression.return_type == "vector" and len(vector_basis) < size:
            vector_basis.append(expression)
        if expression.return_type == "matrix" and len(matrix_basis) < size:
            matrix_basis.append(expression)

        if len(vector_basis) >= size and len(matrix_basis) >= size: break

    basis_cache[basis_key] = (vector_basis, matrix_basis)
    return basis_cache[basis_key]


if __name__ == '__main__':
    vector_basis, matrix_basis = construct_basis([], ["R", "V1","V2"], size=40000, weighted=True)
