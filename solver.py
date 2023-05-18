from z3 import *

slv = Solver()

recent_model = None
def get_recent_model():
    global recent_model
    return recent_model

def push_solver():
    slv.push()
def pop_solver():
    slv.pop()
def set_solver(s):
    global slv
    slv = s
def get_solver():
    global slv
    return slv

def exactly_one(bs):
    constrain(Or(bs))
    for i, b in enumerate(bs):
        for b2 in bs[i+1:]:
            constrain(Not(And(b,b2)))
    return bs

def constrain(constraint): slv.add(constraint)

def solve():
    if "unsat" in str(slv.check()):
        return None
    else:
        return slv.model()


class SLC:
    """Straight Line Code"""
    def __init__(self, inputs, lines, components):

        self.inputs = inputs
        
        self.op = [ exactly_one([ FreshBool() for _ in components ]) for _ in range(lines) ]
        self.left = [ exactly_one([ FreshBool() for _ in range(len(inputs)+l) ])
                      for l in range(lines) ]
        self.right = [ exactly_one([ FreshBool() for _ in range(len(inputs)+l) ])
                       for l in range(lines) ]
        self.lines = lines
        self.components = components

    def execute(self, *xs):

        values = [ FreshReal() for _ in range(self.lines) ]

        for ln, v in zip(range(self.lines), values):

            this_left, this_right = FreshReal(), FreshReal()
            for x,l,r in zip(xs, self.left[ln], self.right[ln]):
                constrain(Implies(l, this_left == x))
                constrain(Implies(r, this_right == x))
            for earlier,l,r in zip(values[:ln], self.left[ln][len(xs):], self.right[ln][len(xs):]):
                constrain(Implies(l, this_left == earlier))
                constrain(Implies(r, this_right == earlier))

            for (_, k), op_flag in zip(self.components, self.op[ln]):
                constrain(Implies(op_flag, values[ln] == k(this_left, this_right)))

        return values[-1]
        
    def extract(self, model):
        extractions = list(self.inputs)

        for ln in range(self.lines):
            f = None
            for (k, _), op_flag in zip(self.components, self.op[ln]):
                if model[op_flag]:
                    f = k
                    break
            assert f is not None

            l = None
            for l_index, l_flag in enumerate(self.left[ln]):
                if model[l_flag]:
                    l = extractions[l_index]
                    break
            assert l is not None

            r = None
            for r_index, r_flag in enumerate(self.right[ln]):
                if model[r_flag]:
                    r = extractions[r_index]
                    break
            assert r is not None

            extractions.append(f"({l} {f} {r})")
        
        return extractions[-1]
            
        
            
        

    
if __name__ == '__main__':
    x = SLC("xy", 5, [("+", lambda x,y:x+y), ("-", lambda x,y:x-y), ("*", lambda x,y:x*y)])
    constrain(x.execute(1,2) == 1-2*2)
    constrain(x.execute(9,-3) == 9-(-3)*(-3))
    print(slv.check())
    print(x.extract(solve()))
