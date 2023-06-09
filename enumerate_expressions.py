import itertools
import time
import random
import numpy as np
import math

class Timing(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.start
        if isinstance(self.message, str):
            message = self.message
        elif callable(self.message):
            message = self.message(dt)
        else:
            raise ValueError("Timing message should be string function")
        print(f"{message} in {dt:.1f} seconds")


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


class NormCubed(Expression):
    return_type = "real"
    argument_types = ["vector"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"NormCubed({self.x})"

    def pretty_print(self):
        return f"{self.x.pretty_print()}^3"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if isinstance(x, np.ndarray):
            return (np.sum(x * x, -1))**(3/2)
        return ((x * x).sum(-1).unsqueeze(-1))**(3/2)

    def arguments(self): return [self.x]


class NormForth(Expression):
    return_type = "real"
    argument_types = ["vector"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"NormForth({self.x})"

    def pretty_print(self):
        return f"{self.x.pretty_print()}^4"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if isinstance(x, np.ndarray):
            return (np.sum(x * x, -1))**(2)
        return ((x * x).sum(-1).unsqueeze(-1))**(2)

    def arguments(self): return [self.x]


class NormFifth(Expression):
    return_type = "real"
    argument_types = ["vector"]

    def __init__(self, x):
        self.x = x

    def __str__(self):
        return f"NormFifth({self.x})"

    def pretty_print(self):
        return f"{self.x.pretty_print()}^5"

    def evaluate(self, environment):
        x = self.x.evaluate(environment)
        if isinstance(x, np.ndarray):
            return (np.sum(x * x, -1))**(5/2)
        return ((x * x).sum(-1).unsqueeze(-1))**(5/2)

    def arguments(self): return [self.x]


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


def expr_structure(expr):
    '''
    Helps disambiguate some of the pretty print structures
    '''
    if type(expr) == Vector or type(expr) == Real:
        return type(expr).__name__ + '(' + str(expr.name) + ')'

    # print class name, parenthesis, and then recurse on the argumentsj
    return (type(expr).__name__
            + '('
            + ', '.join([expr_structure(arg) for arg in expr.arguments()])
            + ')')


GOAL_EXPRS = []
def weighted_bottom_up_generator(operators, constants, inputs, cost_dict=None):
    """
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

    for op in operators:
        if op not in cost_dict:
            cost_dict[op] = 1
            print(f"warning: no cost provided for {op}. using cost 1")

    for v in variables_and_constants:
        if type(v) not in cost_dict:
            cost_dict[type(v)] = 1
            print(f"warning: no cost provided for {v}. using cost 1")

    assert all(isinstance(cost, int) for cost in cost_dict.values()), 'costs must be integral'

    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size
    observed_values = set()

    dim = max([len(v) for v in inputs[0].values()])
    check_goal = 'V1' in inputs[0] and dim == 3
    if check_goal:
        R = Vector('R')
        V1 = Vector('V1')

        goal1_expr = Skew(ScaleInverse(V1, Inner(R, Scale(Length(R), R))))
        goal2_expr = ScaleInverse(Outer(Cross(V1, R), Hat(R)), Times(Inner(R, R), Inner(R, R)))
        # goal1_expr = Skew(ScaleInverse(V1, NormCubed(R)))
        # goal2_expr = ScaleInverse(Outer(Cross(V1, R), R), NormFifth(R))
        goal_exprs = [goal1_expr, goal2_expr]

        goal_vals = [np.array([goal_expr.evaluate(input) for input in inputs]) for goal_expr in goal_exprs]
        goal_vals = [goal_val / np.linalg.norm(goal_val) for goal_val in goal_vals]
        goal_v1s = [np.around(goal_val*100, decimals=5).tobytes() for goal_val in goal_vals]


    enumerated_expressions = {}
    def record_new_expression(expression, cost):
        """Returns True iff the semantics of this expression has never been seen before"""
        nonlocal inputs, observed_values

        # print(expression.pretty_print(), '\t', expr_structure(expression))
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

        # things we hash.tobytes(
        v1 = np.around(valuation*100, decimals=5).tobytes()
        v2 = np.around(-valuation*100, decimals=5).tobytes()

        # is this something we have not seen before?
        if v1 not in observed_values and v2 not in observed_values:
            observed_values.add(v1)

            if check_goal:
                for i, goal_v1 in enumerate(goal_v1s):
                    if v1 == goal_v1 or v2 == goal_v1:
                        print(f'found term {i+1}: {expression.pretty_print()}')
                        print(expr_structure(expression))
                        GOAL_EXPRS.append(expression.pretty_print())

            # we have some new behavior
            key = (expression.__class__.return_type, cost)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, (v1, len(observed_values))))
            return True

        return False

    lvl = 0
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
                            # if len(observed_values) % 1000 == 0:
                                # print(f'{len(observed_values)} expressions')
                            yield new_expression

        lvl += 1


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


def fit_pcfg(expressions, operators, pseudocount=1):
    '''
     Returns a cost dict for the PCFG that is a good fit for the given expressions.
     Infers variables from those present in expressions.

     pseudocount is added to the count of each operator, to avoid zero probabilities.
    '''

    def get_variables(expression):
        if isinstance(expression, Vector) or isinstance(expression, Real):
            return {type(expression)}
        else:
            return set().union(*[get_variables(arg) for arg in expression.arguments()])

    variables = set().union(*[get_variables(expr) for expr in expressions])
    vars_and_ops = list(variables) + operators

    # the "counts" are the number of times that expression occurs
    # some expressions might occur more than others because that type is more common or something.
    # as a start, normalize by the number of times expressions of that type occur
    counts = {op: pseudocount for op in vars_and_ops}
    type_counts = {op.return_type: 0 for op in vars_and_ops}

    # add pseudocount to the denominator for each op of that type
    for op in vars_and_ops:
        type_counts[op.return_type] += pseudocount

    def calc_counts(expressions):
        for expr in expressions:
            counts[type(expr)] += 1
            calc_counts(expr.arguments())

    def calc_type_counts(expressions):
        for expr in expressions:
            type_counts[expr.return_type] += 1
            calc_type_counts(expr.arguments())

    calc_counts(expressions)
    calc_type_counts(expressions)

    # print(f'{counts=}')
    # print(f'{type_counts=}')

    # probability = count / type_count
    # cost = rounded negative log of probability
    cost_dict = {op: -math.log(count / type_counts[op.return_type]) for op, count in counts.items()}
    # directly rounding is too coarse, so instead round to nearest 1/10th.
    # then multiply by 10 so costs are still integers
    cost_dict = {op: round(cost * 10) for op, cost in cost_dict.items()}

    # don't allow cost zero... add one to each
    cost_dict = {op: cost + 1 for op, cost in cost_dict.items()}

    return cost_dict


def infer_variables(inputs):
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)

    variables = list({make_variable(variable_name, variable_value)
                      for input in inputs
                      for variable_name, variable_value in input.items() })
    # since inputs are randomly generated, this ensures a consistent order
    # so that the synthesis order is deterministic
    variables = sorted(variables, key=lambda v: v.name)
    return variables

    # goal1_expr = Skew(ScaleInverse(V1, Inner(R, Scale(Length(R), R))))
    # goal2_expr = ScaleInverse(Outer(Cross(V1, R), Hat(R)), Times(Inner(R, R), Inner(R, R)))

    # goal1_expr = Skew(ScaleInverse(V1, NormCubed(R))
    # goal2_expr = ScaleInverse(Outer(Cross(V1, R), R), NormFifth(R))

def get_operators(dimension=3):

    operators = [
        # Divide,
        Outer,
        ScaleInverse,

        Hat,
        Length,
        Scale,
        Inner,
        Times,

        # NormCubed,
        # NormForth,
        # NormFifth,
    ]

    if dimension == 3: operators.extend([
        Skew,
        Cross,
    ])
    return operators

basis_cache = {}
def construct_basis(reals, vectors, size, dimension=3, cost_dict=None):
    basis_key = (tuple(reals), tuple(vectors), size, dimension)
    if cost_dict is None and basis_key in basis_cache: return basis_cache[basis_key]

    operators = get_operators(dimension)

    def random_input():
        d = {}
        for nm in reals:
            d[nm] = random.random()*10
        for nm in vectors:
            d[nm] = np.random.random(dimension)*10-5
        return d

    inputs = [random_input() for _ in range(10) ]

    vector_basis = []
    matrix_basis = []

    variables = infer_variables(inputs)
    constants = []
    variables_and_constants = variables + constants

    if any(v.name == 'V1' for v in variables):
        V1 = [v for v in variables if v.name == 'V1'][0]
        R = [r for r in variables if r.name == 'R'][0]

        handcrafted_exprs = [
            # Skew(ScaleInverse(V1, Inner(R, Scale(Length(R), R)))),
            # Outer(ScaleInverse(Cross(V1, Hat(R)), Times(Inner(R, R), Inner(R, R))), R),
        ]

        for expression in handcrafted_exprs:
            print('adding handcrafted expr to enumerated exprs: ', expression.pretty_print())
            if expression.return_type == "vector":
                vector_basis.append(expression)
            if expression.return_type == "matrix":
                matrix_basis.append(expression)

    with Timing("weighted bottom up generator"):
        for expression in weighted_bottom_up_generator(operators, constants,
                                                      inputs, cost_dict=cost_dict):
            if expression.return_type == "vector" and len(vector_basis) < size:
                vector_basis.append(expression)

            if expression.return_type == "matrix" and len(matrix_basis) < size:
                matrix_basis.append(expression)
                # if len(GOAL_EXPRS) == 2:
                    # print(f'{len(matrix_basis)} matrix basis terms to find goals')
                    # print(f'{len(vector_basis)} vector basis terms')
                    # assert False


            if len(vector_basis) >= size and len(matrix_basis) >= size: break

    basis_cache[basis_key] = (vector_basis, matrix_basis)
    return basis_cache[basis_key]


def dipole_cost_dict():
    operators = get_operators()
    cost_dict = {op: 100 for op in operators}

    R = Vector('R')
    V1 = Vector('V1')

    # when we have norm forth and norm cubed
    if NormFifth in operators and NormCubed in operators:
        p1 = Skew(ScaleInverse(V1, NormCubed(R)))
        p2 = Outer(Cross(R, V1), ScaleInverse(R, NormFifth(R)))
    # when we only have norm fifth
    elif NormFifth in operators and NormCubed not in operators and NormForth not in operators:
        p1 = Skew(ScaleInverse(V1, Length(Scale(Inner(R, R), R))))
        p2 = Outer(Cross(R, V1), ScaleInverse(R, NormFifth(R)))
    # when we only have norm cubed
    elif NormCubed in operators and NormFifth not in operators and NormForth not in operators:
        p1 = Skew(ScaleInverse(V1, NormCubed(R)))
        p2 = Outer(Cross(V1, Hat(R)), ScaleInverse(Hat(R), NormCubed(R)))
    # when we have no norms
    elif NormCubed not in operators and NormFifth not in operators and NormForth not in operators:
        p1 = ScaleInverse(Outer(Cross(V1, R), Hat(R)), Times(Inner(R, R), Inner(R, R)))
        # p2 = Skew(ScaleInverse(V1, Length(Scale(Inner(R, R), R))))
        p2 = None
    else:
        raise ValueError('unknown operator combination')

    exprs = [p1 for _ in range(10)]
    if p2 is not None:
        exprs += [p2 for _ in range(10)]

    cost_dict = fit_pcfg(exprs, get_operators(), pseudocount=0.40)

    # cost_dict[Vector] = 5
    cost_dict[Outer] = 2
    # cost_dict[ScaleInverse] = 24
    # cost_dict[Hat] = 24
    # cost_dict[Length] = 45
    # cost_dict[Scale] = 57
    # cost_dict[Inner] = 5
    # cost_dict[Times] = 12
    # cost_dict[Skew] = 34
    # cost_dict[Cross] = 24

    for op in cost_dict:
        print(f'{op}\t{op.return_type}\t\t{cost_dict[op]}')

    return cost_dict

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    # vector_basis, matrix_basis = construct_basis([], ["R", "V1","V2"], size=100000, cost_dict=dipole_cost_dict(get_operators()))
    vector_basis, matrix_basis = construct_basis([], ["R", "V1","V2"], size=2000, cost_dict=None)
    print(len(vector_basis), len(matrix_basis))
