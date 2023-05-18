import z3
from solver import SLC, constrain, solve


class ForceLearner():
    def __init__(self, n_latent, n_lines):
        self.n_latent = n_latent
        self.n_lines = n_lines

    def fit(self, acceleration_laws):

        basis_functions = list({(bf, len({i,j}))
                                for terms in acceleration_laws
                                for bf, i, j, *rest in terms })

        masses = [z3.Real(f"M_{i+1}") for i in range(len(acceleration_laws)) ]
        latent = [ [z3.Real(f"L{j+1}_{i+1}") for j in range(self.n_latent) ]
                   for i in range(len(acceleration_laws)) ]

        components = [("+", lambda x,y:x+y), ("-", lambda x,y:x-y),
                      ("*", lambda x,y:x*y), ("/", lambda x,y:x/y)]

        for i in range(len(acceleration_laws)):
             if False and i == 0:
                 constrain(masses[i] == 1)
             else:
                 constrain(masses[i] > 0)

        coefficient_programs = {}
        for b, n_particles in basis_functions:
            assert b.return_type == "vector", "latent vectors not yet supported"
            
            if n_particles == 1:
                inputs = ["M"] + [f"L{l+1}" for l in range(self.n_latent) ]
            else:
                inputs = ["M1", "M2"] + [f"L{l+1}_self" for l in range(self.n_latent) ] + [f"L{l+1}_other" for l in range(self.n_latent) ]
                
            program = SLC(inputs, self.n_lines, components)

            # figure out the examples for this program
            for i, terms in enumerate(acceleration_laws):
                for other_b, other_i, j, coefficient in terms:
                    assert other_i == i
                    if other_b != b: continue

                    if n_particles == 1:
                        inputs = [masses[i]] + latent[i]
                    else:
                        inputs = [masses[i],masses[j]] + latent[i] + latent[j]

                    predicted_force_coefficient = program.execute(*inputs)
                    predicted_acceleration_coefficient = predicted_force_coefficient/masses[i]

                    constrain(predicted_acceleration_coefficient < coefficient+1e-2)
                    constrain(predicted_acceleration_coefficient > coefficient-1e-2)
            
            coefficient_programs[b]=program

        model = solve()

        for b, program in coefficient_programs.items():
            print("Coefficient for", b.pretty_print(), "can be predicted using:")
            print(program.extract(model))
            print()

        print("\t\t\tmass\tlatents")
        for i in range(len(acceleration_laws)):
            text = f"particle {i}\t\t"+"\t".join(str(model[unobserved]) for unobserved in [masses[i]] + latent[i] )
            print(text)
            
        
