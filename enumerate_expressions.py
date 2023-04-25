import itertools
import random
import numpy as np


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
    
def bottom_up_generator(global_bound, operators, constants, input_outputs):
    """
    global_bound: int. an upper bound on the size of expression
    operators: list of classes, such as [Times, If, ...]
    constants: list of possible leaves in syntax tree, such as [Number(1)]. Variables can also be leaves, but these are automatically inferred from `input_outputs`
    input_outputs: list of tuples of environment (the input) and desired output, such as [({'x': 5}, 6), ({'x': 1}, 2)]
    yields: sequence of programs, ordered by expression size, which are semantically distinct on the input examples
    """

    # variables and constants should be treated the same, because they are both leaves in syntax trees
    # after computing `variables_and_constants`, you should no longer refer to `constants`. express everything in terms of `variables_and_constants`
    # `make_variable` is just a helper function for making variables that smartly wraps the variable name in the correct class depending on the type of the variable
    def make_variable(variable_name, variable_value):
        if isinstance(variable_value, float): return Real(variable_name)
        if isinstance(variable_value, np.ndarray): return Vector(variable_name)
        
    variables = list({make_variable(variable_name, variable_value)
                      for inputs, outputs in input_outputs
                      for variable_name, variable_value in inputs.items() })
    variables_and_constants = constants + variables

    # suggested data structure (you don't have to use this if you don't want):
    # a mapping from a tuple of (type, expression_size) to all of the possible values that can be computed of that type using an expression of that size
#starting
    observed_values = set()

    enumerated_expressions = {}
    def record_new_expression(expression, size):
        """Returns True iff the semantics of this expression has never been seen before"""
        nonlocal input_outputs, observed_values

        valuation = tuple(expression.evaluate(input) for input, output in input_outputs)

        # discard all zeros?
        #if all( np.max(np.abs(v)) < 1e-5 for v in valuation ):
        #    return False # bad expression

        # calculate what values are produced on these inputs
        values = tuple(str(v) for v in valuation)

        # is this something we have not seen before?
        if values not in observed_values: 
            observed_values.add(values)

            # we have some new behavior
            key = (expression.__class__.return_type, size)

            if key not in enumerated_expressions:
                enumerated_expressions[key] = []

            enumerated_expressions[key].append( (expression, values) )

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
#ending
    assert False, "implement as part of homework"

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

basis_cache = {}
def construct_basis(reals, vectors, size, dimension=3):
    basis_key = (tuple(reals), tuple(vectors), size, dimension)
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
    
    input_outputs = [(random_input(), None)
                     for _ in range(10) ]
    count = 0
    vector_basis = []
    matrix_basis = []
    for expression in bottom_up_generator(10, operators, constants, input_outputs):
        if expression.return_type == "vector" and len(vector_basis) < size:
            vector_basis.append(expression)
        if expression.return_type == "matrix" and len(matrix_basis) < size:
            matrix_basis.append(expression)
        
        if len(vector_basis) >= size and len(matrix_basis) >= size: break

    basis_cache[basis_key] = (vector_basis, matrix_basis)
    return basis_cache[basis_key]
    
if __name__ == '__main__':
    for e in construct_basis([],#"R", "V1","V2"
                             ["R", "V1","V2"],
                             10000)[1]:
        print(e.pretty_print())
