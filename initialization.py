# import the nanowire_hamiltonian package
import importlib

import numpy as np

import nanowire_hamiltonian_class as nh
import potential_barrier_class as pb

# Get physical values from scipy, hbar and m_e

# Now we want to reload the modules pb and nh
importlib.reload(pb)
importlib.reload(nh)


def potential_function(x, x0, sigma):
    return np.exp((x - x0)**2 / (2 * sigma**2))


def initialize_standard_hamiltonian():
    # define the potential barrier values
    x0 = 0.0
    barrier_height = 10.0
    sigma = 10.0

    # define the hamiltonian parameters, energies in meV, length in m, mass in m_e
    alpha = 50.0
    zeeman = 2.0
    chem_pot = 5.0
    sc_gap = 0.5
    eff_mass = 0.015
    nw_length = 4e+3
    grid_points = 100

    # initialize the potential barrier
    poti = pb.potential_barrier(potential_function, x0, sigma, barrier_height)

    # initialize the hamiltonian
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, grid_points, poti)
    hami.greet()
    hami.build_hamiltonian(0.0)


def main():
    initialize_standard_hamiltonian()


if __name__ == '__main__':
    main()
