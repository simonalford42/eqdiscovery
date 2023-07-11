from animate import animate
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import assert_equal, assert_shape



def simulate_learned_laws(init_x, init_v, laws, steps=100, dt=0.01):
    '''
    simulate a system of particles with given initial positions and velocities from the acceleration laws learned.
    '''
    print(f'{steps=}')
    n_particles = init_x.shape[0]
    n_dims = init_x.shape[1]
    assert init_v.shape == (n_particles, n_dims)
    assert n_dims == 2, 'only 2D for now'

    # (n_particles, n_dims)
    xs = [init_x]
    v = init_v

    for _ in range(steps):
        # each particle has its own set of terms for its acceleration law
        new_x = []
        new_v = []
        x = xs[-1]

        for particle, terms in enumerate(laws):
            a = np.zeros(n_dims)
            for (expr, i, j, c) in terms:
                # goal: get acceleration contribution from this law
                # expr contains variables such as V, V1, V2, R, R1, R2
                # based on acceleration show laws:
                # 1. provide V_i as V to the environment
                # 2. provide V_i as V1 and V_j as V2 to the environment
                # 3. provide R_{ij} as R to the environment
                # 4. everything else is provided "as is" to the environment
                env = {'V': v[i],
                       'V1': v[i], 'V2': v[j],
                       'R': x[i]-x[j]}
                val = expr.evaluate(env)
                if expr.return_type == 'vector':
                    a += val
                else:
                    assert expr.return_type == 'matrix'
                    # same thing as val @ c
                    # for the matrix result, m[d, u], d is the dimension axis and u is the coefficient axis
                    a += np.einsum('ij, j -> i', val, c[:n_dims])

            dv = a*dt
            dx = v[particle]*dt
            new_x.append(x[particle]+dx)
            new_v.append(v[particle]+dv)

        xs.append(np.stack(new_x))
        v = np.stack(new_v)

    return np.stack(xs)


def animate_simon(x, name):
    from matplotlib import pyplot as plt
    import os
    T = len(x)
    n_particles = x.shape[1]
    assert_shape(x, (T, n_particles, 2))  # T, n_particles, dims
    # minx, maxx = 0, WIDTH
    # miny, maxy = 0, HEIGHT
    buffer = 0.1
    minx, maxx = np.min(x[:, :, 0]), np.max(x[:, :, 0])
    miny, maxy = np.min(x[:, :, 1]), np.max(x[:, :, 1])
    bufferx = buffer * (maxx - minx)
    minx, maxx = minx - bufferx, maxx + bufferx
    buffery = buffer * (maxy - miny)
    miny, maxy = miny - buffery, maxy + buffery

    # remove all images in /tmp/boids
    os.system(f"rm /tmp/{name}_*.png")

    print('animating...')
    for t in range(T):
        plt.figure()
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])

        plt.scatter(x[t, :, 0], x[t, :, 1])
        plt.savefig(f"/tmp/{name}_{t:03d}.png")
        plt.close()
    print('done, now converting images to gif... ')

    os.system(f"gm convert -delay 5 -loop 0 /tmp/{name}_*.png {name}.gif")
    print('saved gif at ', name + '.gif')

def simulate_elastic_pendulum():
    m = 10
    k = 1
    L = 10
    g = 2

    T = 100
    dt = 0.001

    init_pos = np.array([7, -7])
    init_vel = np.array([0, 0])

    def cartesian_to_polar(x, y):
        return np.arctan2(y, x), (x**2 + y**2)**0.5

    x = [init_pos]
    v = [init_vel]
    f = []
    a = []

    for _ in range(int(T/dt)):
        theta, length = cartesian_to_polar(*x[-1])
        # spring
        # force = -k*(length-L) * np.array([np.cos(theta), np.sin(theta)])
        force = 0
        # gravity
        force = force - m*g*np.array([0,1])

        dv = force/m*dt

        dx = v[-1]*dt

        f.append(force)
        a.append(force / m)
        x.append(x[-1]+dx)
        v.append(v[-1]+dv)

    x, v, f, a = [np.stack(arr)[:, None, :][::100] for arr in [x[:len(f)], v[:len(f)], f, a]]
    # return x, v, f, a
    # make a dummy particle for the center of the spring
    x2, v2, f2, a2 = np.zeros_like(x), np.zeros_like(v), np.zeros_like(f), np.zeros_like(a)

    x, v, f, a = [np.concatenate([a, b], axis=1) for (a, b) in [(x, x2), (v, v2), (f, f2), (a, a2)]]

    return x, v, f, a



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

