import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class NormalizedLinear(nn.Module):
    """weight matrix constrained to lie on the simplex"""
    def __init__(self, in_features, out_features,
                 device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self._weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))

    @property
    def weight(self):
        return torch.softmax(self._weight, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """input @ (W^T)"""
        return F.linear(input, self.weight, None)

    def sparse_regularizer(self):        
        return -(torch.softmax(self._weight, -1)*F.log_softmax(self._weight, -1)).sum()
        

class RegularLinear(nn.Module):
    """Just your every day regular weight matrix"""
    def __init__(self, in_features, out_features,
                 device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self._weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def sparse_regularizer(self):
        return self.weight.abs().sum()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))

    @property
    def weight(self):
        return self._weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """input @ (W^T)"""
        return F.linear(input, self.weight, None)    
        
    
class GSMLinear(nn.Module):
    """
    weight matrix with bounded magnitude and masked by GSM
    out[n] = \sum_m in[m] * tanh(W[n,m]) * Mask[n,m]
    Mask[n,:] = GSM(Logits[n,:])
    """
    def __init__(self, in_features, out_features,
                 device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self._weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self._logits = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.temperature = 1

    def sparse_regularizer(self):
        return 0

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))
        init.kaiming_uniform_(self._logits, a=math.sqrt(5))

    @property
    def weight(self):
        return 2 * nn.Tanh()(self._weight) * F.gumbel_softmax(self._logits, hard=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        W = 2*nn.Tanh()(self.weight)
        L = self._logits
        
        #input:[b1,b2,...,in_features]
        #L:[b1,b2,...,out_features,in_features]
        while len(L.shape)-1 < len(input.shape):
            L = L.unsqueeze(0)
            W = W.unsqueeze(0)
            
        L = L.repeat([b for b in input.shape[:-1] ]+[1,1])
        
        L = F.gumbel_softmax(L, hard=False, tau=self.temperature)

        W = W*L

        y = torch.matmul(W, input.unsqueeze(-1)).squeeze(-1)
        return y

class GSM_ones_only(nn.Module):
    """
        out[n] = \sum_m in[m] * Mask[n,m]
    Mask[n,:] = GSM(Logits[n,:])
    """
    def __init__(self, in_features, out_features,
                 device=None, dtype=None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self._logits = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.temperature = 1

    def sparse_regularizer(self):
        return 0

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._logits, a=math.sqrt(5))

    @property
    def weight(self):
        return F.gumbel_softmax(self._logits, hard=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        L = self._logits
        
        #input:[b1,b2,...,in_features]
        #L:[b1,b2,...,out_features,in_features]
        while len(L.shape)-1 < len(input.shape):
            L = L.unsqueeze(0)
                        
        L = L.repeat([b for b in input.shape[:-1] ]+[1,1])
        
        L = F.gumbel_softmax(L, hard=False, tau=self.temperature)

        y = torch.matmul(L, input.unsqueeze(-1)).squeeze(-1)
        return y    
        

class VectorLayer(nn.Module):
    def __init__(self, inputs, operators):
        """
        inputs: mapping from dimension to count
        operators: mapping from name to (count, type, implementation)
        """
        super().__init__()

        # remove any operators which do not have all of their argument
        self.operators = {name: (count, tp, f)
                          for name, (count, tp, f) in operators.items()
                          if all(a in inputs for a in tp[:-1] )}

        output_types = {tp[-1] for _, tp, _ in self.operators.values() }
        self.outputs = {tp: sum( count for count, tp2, _ in self.operators.values() if tp2[-1] == tp)
                        for tp in output_types }
        self.inputs = inputs

        def get_projector(name):
            if False:
                if name == "hat":
                    return GSM_ones_only
                else:
                    return GSMLinear
        
            if name == "hat":
                return NormalizedLinear
            else:
                return RegularLinear
            
        self.projections = \
              nn.ModuleDict({name: nn.ModuleList(get_projector(name)(inputs[d], count)
                                                 for d in tp[:-1] )
                             for name, (count, tp, _) in self.operators.items() })

    def set_temperature(self, t):
        for modules in self.projections.values():
             for module in modules:
                 module.temperature = t
        
    def forward(self, x):
        for dimension, count in self.inputs.items():
            assert x[dimension].shape[-1] == dimension
            assert x[dimension].shape[-2] == count, f"{count} {x[dimension].shape}"

        y = {tp: [] for tp in self.outputs}

        for op in sorted(self.projections):
            count,tp,f = self.operators[op]
            projectors = self.projections[op]

            input_types, output_type = tp[:-1], tp[-1]

            f_inputs = []
            for d, pr in zip(input_types, projectors):
                # x[d] : b, num_inputs_of_dimension_d, d
                # pr : num_inputs_of_dimension_d -> count
                # we want one of the inputs to f
                # should be of shape [b, count, d]

                next_input = pr(x[d].transpose(-1,-2)).transpose(-1,-2)
                assert next_input.shape[0] == x[d].shape[0]
                assert next_input.shape[1] == count
                assert next_input.shape[2] == d
                
                f_inputs.append(next_input)

            #print("about to run", op, "on the inputs", [ni.shape for ni in f_inputs])
            y[output_type].append(f(*f_inputs))
        
        y = {d: torch.cat(ys,1) for d, ys in y.items() }
        for d in self.outputs:
            assert y[d].shape[-1] == d
            assert y[d].shape[-2] == self.outputs[d]
        return y

    def extract(self, input_names):
        
        def linear_combination(inputs, coefficients):
            combination=[]
            for iname, coefficient in zip(inputs, coefficients): #linear_layer.weight[i]):
                c = coefficient.item()
                if c > 1e-4: combination.append("+%.04f%s"%(c, iname))
                if c < -1e-4: combination.append("%.04f%s"%(c, iname))
            return ''.join(combination)

        outputs = {d: [] for d in self.outputs.keys() }

        for op in sorted(self.projections):
            count,tp,f = self.operators[op]
            projectors = self.projections[op]

            input_types, output_type = tp[:-1], tp[-1]

            for i in range(count):
                arguments = ", ".join(linear_combination(input_names[input_types[a]],
                                                         projectors[a].weight[i])
                                      for a in range(len(input_types)) )
                outputs[output_type].append(f"{op}({arguments})")

        return outputs

    def simplified_equation(self, input_names, coordinate_frame):
        import sympy
        import sympy.physics.vector as sv

        def identity(z): return z
        
        symbolic_backend =\
             {"id1": identity, "id2": identity, "identity1": identity, "identity2": identity,
              "dot-product": lambda u,v: u.dot(v), 
              "scale-vector": lambda u,v: u*v,
              "*": lambda u,v: u*v,
              "hat": lambda u: u.normalize()}

        construct_zero = {1: 0.,
                          2: coordinate_frame.i*0,
                          3: coordinate_frame.i*0}
        
        def linear_combination(inputs, coefficients, rank):
            combination=construct_zero[rank]
            for iname, coefficient in zip(inputs, coefficients): #linear_layer.weight[i]):
                c = coefficient.item()
                if abs(c) > 1e-4: combination = combination + c*iname
            
            return combination

        outputs = {d: [] for d in self.outputs.keys() }

        for op in sorted(self.projections):
            count,tp,f = self.operators[op]
            projectors = self.projections[op]

            input_types, output_type = tp[:-1], tp[-1]

            for i in range(count):
                arguments = [linear_combination(input_names[input_types[a]],
                                                projectors[a].weight[i],
                                                input_types[a])
                             for a in range(len(input_types)) ]
                
                outputs[output_type].append(symbolic_backend[op](*arguments))

        return outputs
        
        

class VectorNetwork(nn.Module):
    def __init__(self, inputs, outputs, operators, layers):
        super().__init__()

        self.outputs, self.inputs = outputs, inputs

        ls = []
        for _ in range(layers):
            ls.append(VectorLayer(inputs,operators))
            inputs = ls[-1].outputs

        # last layer is linear, identity activation
        output_operators = {f"identity{d}": (count, (d,d), lambda z: z)
                            for d, count in outputs.items() }
        
        self.model = nn.Sequential(*ls+[VectorLayer(inputs, output_operators)])

    def forward(self, x):
        return self.model(x)

    def l1(self):
        return sum(p.sparse_regularizer()
                   for l in self.model
                   for ps in l.projections.values()
                   for p in ps )
    def fancy_regularizer(self):
        assert False
        def f(x):
            return (x<0)*1*(x.abs()) + (0<x)*(x<0.5)*x*1 + (1-x)*1*(0.5<x)*(x<1) + (x>=1)*1*(x-1)
        return sum( f(w).sum()
                   for w in self.parameters() )

    def clamp_parameters(self):
        assert False
        with torch.no_grad():
            for wm in self.parameters():
                wm.data *= 1*(wm.abs()>1e-2)

    def set_temperature(self, t):
        for l in range(len(self.model)):
            self.model[l].set_temperature(t)
    
    def extract(self, input_names, output_names=None):
        extraction = []
        for l in range(len(self.model)):
            expressions = self.model[l].extract(input_names)

            if l < len(self.model)-1:
                vn = "z"
            else:
                vn = "y"
            
            extraction.extend([f"{vn}{d}_{l}_{j} = {e}"
                               for d in sorted(expressions) 
                               for j, e in enumerate(expressions[d]) ])
            input_names = {d: [f"{vn}{d}_{l}_{j}" for j, e in enumerate(expressions[d])]
                           for d in sorted(expressions) }

        return "\n".join(extraction)

    def simplified_equation(self, input_names):
        from sympy import Symbol
        from sympy.vector import CoordSys3D
        N = CoordSys3D('N')

        def make_symbol(n, rank):
            if rank == 1:
                return Symbol(n)
            if rank == 2:
                return Symbol(n+"_x")*N.i+Symbol(n+"_y")*N.j
            if rank == 3:
                return Symbol(n+"_x")*N.i+Symbol(n+"_y")*N.j+Symbol(n+"_z")*N.k
            assert False
                            
        expressions = { rank: [make_symbol(n, rank) for n in names ]
                        for rank, names in input_names.items() }

        for l in range(len(self.model)):
            expressions = self.model[l].simplified_equation(expressions, N)

        return expressions

class Sindy(nn.Module):
    def __init__(self, reals, vectors, size):
        super().__init__()

        from enumerate_expressions import construct_basis
        
        self.expressions = construct_basis(reals, vectors, size)
        print("\n".join([e.pretty_print() for e in self.expressions]))
        #assert any([e.pretty_print() == "(* M1 (* (/ M2 (dp R R)) (hat R)))" for e in self.expressions ])
        #self.expressions = [e for e in self.expressions if e.pretty_print() == "(* M1 (* (/ M2 (dp R R)) (hat R)))"]
        
        self.model = nn.Linear(len(self.expressions), 1, bias=False)

    def forward(self, variables):
        x = torch.stack([ e.evaluate(variables) for e in self.expressions], -1)
        
        x = x.clamp(min=-100, max=+100)
        
        return self.model(x).squeeze(-1)

    def l1(self):
        return self.model.weight.abs().sum()*0

    

    def set_temperature(self, t):
        pass
        
    
    def extract(self):
        terms = [ (self.expressions[i], self.model.weight[0,i].detach().cpu().numpy())
                  for i in range(len(self.expressions)) ]
        terms.sort(key = lambda ew: -abs(ew[1]))

        extraction = [ f"{w}\t{e.pretty_print()}"
                       for e,w in terms ]

        return "\n".join(extraction)

    

class EquationLayer(nn.Module):
    def __init__(self, inputs, operators):
        """
        unary: mapping from the name of a unary function to (count, callback)
        binary is similar
        """
        super().__init__()

        unary, binary = {name: (count, callback)
                         for name, (count, callback) in operators.items()
                         if len(callback.__code__.co_varnames) == 1}, \
                        {name: (count, callback)
                         for name, (count, callback) in operators.items()
                         if len(callback.__code__.co_varnames) == 2}

        self.unary, self.binary = unary, binary

        self.unary_projections = \
                nn.ModuleDict({name: nn.Linear(inputs, count)
                               for name, (count, _) in unary.items() })
        self.binary_projections = \
                nn.ModuleDict({name: nn.Linear(inputs, 2*count)
                               for name, (count, _) in binary.items() })

        self.outputs = sum(count for (count, _) in operators.values())

    def forward(self, x):
        """x: [b,inputs]
        returns: [b,outputs]"""
        outputs = []
        for name in sorted(self.unary):
            _, f = self.unary[name]

            outputs.append(f(self.unary_projections[name](x)))

        for name in sorted(self.binary):
            _, f = self.binary[name]
            i = self.binary_projections[name](x)
            a1, a2 = i[..., :i.shape[-1]//2], i[..., i.shape[-1]//2:]
            
            outputs.append(f(a1, a2))
        
        return torch.concat(outputs, -1)

    def extract(self, input_names):
        
        def linear_combination(linear_layer, i):
            combination=[]
            for iname, coefficient in zip(input_names, linear_layer.weight[i]):
                c = coefficient.item()
                if c > 1e-2: combination.append("+%.02f%s"%(c, iname))
                if c < -1e-2: combination.append("%.02f%s"%(c, iname))

            bias = ""
            b = linear_layer.bias[i].item()
            if b > 1e-2: bias = "+%.02f"%b
            if b < -1e-2: bias = "%.02f"%b

            return f"{''.join(combination)}{bias}"
            
        outputs = []
        for name in sorted(self.unary):
            count = self.unary_projections[name].out_features

            for i in range(count):
                outputs.append(f"{name}({linear_combination(self.unary_projections[name], i)})")

        for name in sorted(self.binary):
            count = self.binary_projections[name].out_features//2            

            for i in range(count):
                outputs.append(f"{name}({linear_combination(self.binary_projections[name], i)}, {linear_combination(self.binary_projections[name], i+count)})")

                
        return outputs

class EquationNetwork(nn.Module):
    def __init__(self, inputs, outputs, layers, operators):
        super().__init__()

        self.outputs = outputs

        ls = []
        for _ in range(layers):
            ls.append(EquationLayer(inputs,operators))
            inputs = ls[-1].outputs

        self.model = nn.Sequential(*ls+[nn.Linear(inputs, outputs)])

    def forward(self, x):
        return self.model(x)

    def l1(self):
        return sum(w.abs().sum()
                   for w in self.parameters() ) #if len(w.shape) == 2)
    def fancy_regularizer(self):
        def f(x):
            return (x<0)*1*(x.abs()) + (0<x)*(x<0.5)*x*1 + (1-x)*1*(0.5<x)*(x<1) + (x>=1)*1*(x-1)
        return sum( f(w).sum()
                   for w in self.parameters() )

    def clamp_parameters(self):
        with torch.no_grad():
            for wm in self.parameters():
                wm *= 1*(wm.abs()>1e-2)

    def extract(self, input_names):
        extraction = []
        for l in range(len(self.model)-1):
            expressions = self.model[l].extract(input_names)
            extraction.extend([f"z{l}_{j} = {e}"
                               for j, e in enumerate(expressions) ])
            input_names = [f"z{l}_{j}" for j, e in enumerate(expressions) ]

        def linear_combination(linear_layer, i):
            combination=[]
            for iname, coefficient in zip(input_names, linear_layer.weight[i]):
                c = coefficient.item()
                if c > 1e-2: combination.append("+%.02f%s"%(c, iname))
                if c < -1e-2: combination.append("%.02f%s"%(c, iname))

            bias = ""
            b = linear_layer.bias[i].item()
            if b > 1e-2: bias = "+%.02f"%b
            if b < -1e-2: bias = "%.02f"%b

            return f"{''.join(combination)}{bias}"

        expressions = [linear_combination(self.model[-1], i)
                       for i in range(self.outputs) ]
        extraction.extend([f"y{j} = {e}" for j, e in enumerate(expressions) ])
        return "\n".join(extraction)
        
        


