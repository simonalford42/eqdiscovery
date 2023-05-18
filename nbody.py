from animate import animate
import os
import numpy as np
import matplotlib.pyplot as plt



def simulate_gravity(masses, positions, velocities, T, steps=None,
                     drag_coefficient=None, drag_exponent=1, 
                     dt=0.01):

    masses, positions, velocities = np.array(masses), np.array(positions), np.array(velocities)
    
    D = positions.shape[1]
    BODIES = masses.shape[0]
    
    x = [positions]
    v = [velocities]
    f = []
    a = []

    for _ in range(int(T/dt)):
        forces = []
        for i in range(BODIES):
            force = np.zeros(D)
            for j in range(BODIES):
                if i != j:
                    displacement = x[-1][i]-x[-1][j]
                    norm2 = np.sum(displacement*displacement, -1)
                    norm = norm2**0.5
                    unit = displacement/norm
                    
                    force = force - masses[i]*masses[j]/norm2 * unit

                if drag_coefficient is not None:
                    if drag_coefficient[i] == float("inf"):
                        force = np.zeros(D)
                    else:
                        this_velocity = v[-1][i]
                        norm2 = np.sum(this_velocity*this_velocity, -1)
                        norm = norm2**0.5
                        if norm > 0:
                            drag_force = -drag_coefficient[i] * (this_velocity/norm) * norm**drag_exponent
                            force = force + drag_force
                    
            forces.append(force)
        forces = np.stack(forces)
        accelerations = forces/masses[:,None]
        
        dv = accelerations*dt
        dx = v[-1]*dt

        f.append(forces)
        a.append(accelerations)
        x.append(x[-1]+dx)
        v.append(v[-1]+dv)

    if steps is not None:
        sampling_frequency = len(f)//steps
        f = f[::sampling_frequency]
        a = a[::sampling_frequency]
        x = x[::sampling_frequency]
        v = v[::sampling_frequency]

    return np.stack(x[:len(f)]), np.stack(v[:len(f)]), np.stack(f), np.stack(a)



def simulate_circular_orbit():
    light_mass = 1
    heavy_mass = 200
    radius = 10
    velocity = (heavy_mass/light_mass/radius)**0.5
    return simulate_gravity([light_mass,heavy_mass],
                            [[radius,0],[0,0]],
                            [[0,velocity],[0,0]],
                            50, steps=1000)

def simulate_drag1():
    return simulate_gravity([10],
                            [[0,0]],
                            [[100,100]],
                            50,
                            drag_coefficient=[1], drag_exponent=1,
                            steps=100)
def simulate_drag2():
    return simulate_gravity([10],
                            [[0,0]],
                            [[100,100]],
                            50,
                            drag_coefficient=[1], drag_exponent=2,
                            steps=1000)

def simulate_drag3():
    return simulate_gravity([1, 20000],
                            [[20,40], [20,-400]],
                            [[0.1,1], [0,0]],
                            50,
                            drag_coefficient=[0.1,float("inf")], drag_exponent=2,
                            steps=50)

def simulate_2_orbits():
    light_mass2 = 2
    light_mass1 = 1
    heavy_mass = 200
    radius1 = 10
    velocity1 = (heavy_mass/light_mass1/radius1)**0.5
    radius2 = 40
    velocity2 = (heavy_mass/light_mass2/radius2)**0.5
    return simulate_gravity([light_mass1,light_mass2,heavy_mass],
                            [[radius1,0],[radius2,0],[0,0]],
                            [[0,velocity1],[0,velocity2],[0,0]],
                            50, steps=100)

def simulate_falling():
    light_mass = 1
    heavy_mass = 2
    radius = 5
    v0 = 0.25
    return simulate_gravity([light_mass,heavy_mass],
                            [[radius,0],[-radius,0]],
                            [[v0,0],[-v0,0]],
                            70, steps=300)





if __name__ == '__main__':
    x,v,f,a = simulate_drag3()
    print(a[:,0,1])
    animate(x)
    
