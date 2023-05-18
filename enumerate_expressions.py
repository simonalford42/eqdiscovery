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
#starting
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

#starting
    def arguments(self): return [self.x, self.y]


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

#starting
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

#starting
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

#starting
    def arguments(self): return [self.x, self.y]

def bottom_up_generator(global_bound, operators, constants, inputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    inputs: list of environments (the input), such as [{'x': 5}, {'x': 1}]
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """

    # variables and constants should be treated the same, because they are both leaves in syntax trees
    # after computing `variables_and_constants`, you should no longer refer to `constants`. express everything in terms of `variables_and_constants`
    # `make_variable` is just a helper function for making variables that smartly wraps the variable name in the correct class depending on the type of the variable
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)

    variables = list({make_variable(variable_name, variable_value)
                      for input in inputs
                      for variable_name, variable_value in input.items() })
    variables_and_constants = constants + variables

    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size

    observed_values = set()

    enumerated_expressions = {}
    def record_new_expression(expression, size):
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

        # things we hash
        v1 = np.around(valuation*100, decimals=5).tobytes()
        v2 = np.around(-valuation*100, decimals=5).tobytes()

        # is this something we have not seen before?
        if v1 not in observed_values and v2 not in observed_values:
            observed_values.add(v1)

            # we have some new behavior
            key = (expression.__class__.return_type, size)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, v1) )

            return True

        return False

    for terminal in variables_and_constants:
        if record_new_expression(terminal, 1): yield terminal

    for target_size in range(2, global_bound + 1): # enumerate programs of increasing size
        for operator in operators:
            partitions = integer_partitions(target_size - 1 - len(operator.argument_types),
                                            len(operator.argument_types))
            for argument_sizes in partitions:
                actual_argument_sizes = [sz+1 for sz in argument_sizes]
                candidate_arguments = [enumerated_expressions.get(type_and_size, [])
                                       for type_and_size in zip(operator.argument_types, actual_argument_sizes)]
                for arguments in itertools.product(*candidate_arguments):
                    new_expression = operator(*[e for e,v in arguments ])
                    if record_new_expression(new_expression, target_size):
                        yield new_expression
    return

def integer_partitions(target_value, number_of_arguments):
    """
    Returns all ways of summing up to `target_value` by adding `number_of_arguments` nonnegative integers
    useful when implementing `bottom_up_generator`:

    Imagine that you are trying to enumerate all expressions of size 10, and you are considering using an operator with 3 arguments.
    So the total size has to be 10, which includes +1 from this operator, as well as 3 other terms from the 3 arguments, which together have to sum to 10.
    Therefore: 10 = 1 + size_of_first_argument + size_of_second_argument + size_of_third_argument
    Also, every argument has to be of size at least one, because you can't have a syntax tree of size 0
    Therefore: 10 = 1 + (1 + size_of_first_argument_minus_one) + (1 + size_of_second_argument_minus_one) + (1 + size_of_third_argument_minus_one)
    So, by algebra:
         10 - 1 - 3 = size_of_first_argument_minus_one + size_of_second_argument_minus_one + size_of_third_argument_minus_one
    where: size_of_first_argument_minus_one >= 0
           size_of_second_argument_minus_one >= 0
           size_of_third_argument_minus_one >= 0
    Therefore: the set of allowed assignments to {size_of_first_argument_minus_one,size_of_second_argument_minus_one,size_of_third_argument_minus_one} is just the integer partitions of (10 - 1 - 3).
    """

    if target_value < 0:
        return []

    if number_of_arguments == 1:
        return [[target_value]]

    return [ [x1] + x2s
             for x1 in range(target_value + 1)
             for x2s in integer_partitions(target_value - x1, number_of_arguments - 1) ]


def pcfg_generator(cost_bound, operators, constants, inputs, pcfg: dict[type, float]):
    """
    Enumerate programs from a probabilistic context-free grammar (PCFG) that are semantically distinct on the input examples.

    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    pcfg: dict mapping Expression subclass to probability of that expression in the PCFG
    yields: sequence of programs, roughly ordered by expression size and cost, which are semantically distinct on the input examples

    Uses approach from https://dl.acm.org/doi/10.1145/3428295
    """

    # variables and constants should be treated the same, because they are both leaves in syntax trees
    # after computing `variables_and_constants`, you should no longer refer to `constants`. express everything in terms of `variables_and_constants`
    # `make_variable` is just a helper function for making variables that smartly wraps the variable name in the correct class depending on the type of the variable
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)

    variables = list({make_variable(variable_name, variable_value)
                      for input in inputs
                      for variable_name, variable_value in input.items() })
    variables_and_constants = constants + variables

    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size

    observed_values = set()

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

        # things we hash
        v1 = np.around(valuation*100, decimals=5).tobytes()
        v2 = np.around(-valuation*100, decimals=5).tobytes()

        # is this something we have not seen before?
        if v1 not in observed_values and v2 not in observed_values:
            observed_values.add(v1)

            # we have some new behavior
            key = (expression.__class__.return_type, cost)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, v1) )

            return True

        return False

    cost_dict = {typ: round(-math.log(p)) for typ, p in pcfg.items()}

    for lvl in range(1, cost_bound + 1):
        for terminal in constants:
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
                        # note: could make use of precomputed evaluation of subtrees when evaluating later
                        new_expression = operator(*[e for e,v in arguments])
                        if record_new_expression(new_expression, lvl):
                            yield new_expression
    return

basis_cache = {}
def construct_basis(reals, vectors, size, dimension=3, use_pcfg=False):
    basis_key = (tuple(reals), tuple(vectors), size, dimension, use_pcfg)
    if basis_key in basis_cache: return basis_cache[basis_key]

    operators = [Hat, Outer, Inner, Divide, Times, Scale, Reciprocal, ScaleInverse, Length]
    if dimension == 3: operators.extend([Skew, Cross])

    constants = []
    def random_input():
        d = {}
        for nm in reals:
            d[nm] = random.random()*10
        for nm in vectors:
            d[nm] = np.random.random(dimension)*10-5
        return d

    inputs = [random_input()
              for _ in range(10) ]
    count = 0
    vector_basis = []
    matrix_basis = []

    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)

    variables = list({make_variable(variable_name, variable_value)
                      for input in inputs
                      for variable_name, variable_value in input.items() })

    pcfg = {op: math.exp(-1) for op in operators} # cost = -log(p), should be 1
    pcfg.update({type(v): math.exp(-1) for v in variables})

    if use_pcfg:
        generator = pcfg_generator(10, operators, variables, inputs, pcfg)
    else:
        generator = bottom_up_generator(10, operators, constants, inputs)

    for expression in generator:
        if expression.return_type == "vector" and len(vector_basis) < size:
            vector_basis.append(expression)
        if expression.return_type == "matrix" and len(matrix_basis) < size:
            matrix_basis.append(expression)

        if len(vector_basis) >= size and len(matrix_basis) >= size: break

    basis_cache[basis_key] = (vector_basis, matrix_basis)
    return basis_cache[basis_key]

if __name__ == '__main__':
    vector_basis, matrix_basis = construct_basis(
        [], ["R", "V1","V2"], size=200, use_pcfg=False)
    pcfg_vector_basis, pcfg_matrix_basis = construct_basis(
        [], ["R", "V1","V2"], size=200, use_pcfg=True)

    print(f'{len(vector_basis)=}')
    print(f'{len(pcfg_vector_basis)=}')
    print(f'{len(matrix_basis)=}')
    print(f'{len(pcfg_matrix_basis)=}')

    for i in range(len(vector_basis)):
        print(vector_basis[i].pretty_print() + '\t\t' + pcfg_vector_basis[i].pretty_print())
