from animate import animate
import os
import numpy as np
import matplotlib.pyplot as plt

def simulate_charge_dipole(masses, positions, velocities, T, steps=None,
                           dt=0.01):
    """https://www.compadre.org/osp/pwa/motionneardipole/"""

    masses, positions, velocities = np.array(masses), np.array(positions), np.array(velocities)

    D = positions.shape[1]
    assert D == 3
    BODIES = masses.shape[0]
    assert BODIES == 2

    x = [positions]
    v = [velocities]
    f = []
    a = []

    for _ in range(int(T/dt)):
        forces = []
        for i in range(BODIES):
            force = np.zeros(D)

            # the dipole
            if i == 1:
                forces.append(force)
                continue

            # the charge
            R = x[-1][i]
            Rl = np.sum(R*R)**0.5
            Rh = R/Rl

            V = v[-1][i]

            M = np.array([1,0,0.])
            B = 1/(Rl**3) * (3*np.sum(M*Rh)*Rh - M)

            force = 10*np.cross(V, B)

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

def simulate_charge_in_uniform_magnetic_field(mass=1, position=[0,0,0.], velocity=[0,1,1.], B=[0.,0.,1.], T=10, steps=1000,
                                              dt=0.001):

    masses, positions, velocities, B = np.array([mass]), np.array([position]), np.array([velocity]), np.array(B)

    D = positions.shape[1]
    assert D == 3
    BODIES = 1

    x = [positions]
    v = [velocities]
    f = []
    a = []

    for _ in range(int(T/dt)):
        forces = []
        for i in range(BODIES):
            force = np.zeros(D)

            V = v[-1][i]
            force = np.cross(V, B)

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



if __name__ == '__main__':
    animate(simulate_charge_in_uniform_magnetic_field(1,
                                                      [0,0,0.],
                                                      [0,1,1.],
                                                      [0.,0.,1.],
                                                      20, steps=100)[0])

    assert False
    animate(simulate_charge_dipole([0.1,1],
                                   [[0,10,0], [0,0,0]],
                                   [[0.05,0,0.1], [0,0,0]],
                                   1000, steps=100, dt=1e-3)[0])
