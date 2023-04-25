# import the nanowire_hamiltonian package
import numpy as np

import nanowire_hamiltonian_class as nh
import potential_barrier_class as pb


def potential_function(x, x0, sigma):
    return np.exp((x - x0)**2 / (2 * sigma**2))


def initialize_standard_hamiltonian():
    # define the potential barrier values
    x0 = 0.0
    sigma = 1.0

    # define the hamiltonian parameters
    alpha = 1.0
    zeeman = 1.0
    chem_pot = 1.0
    eff_mass = 1.0

    # initialize the potential barrier
    poti = pb.potential_barrier(potential_function, x0, sigma)
    print(poti(0.0))

    # initialize the hamiltonian
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, eff_mass, poti)
    hami.greet()
    print(hami.evaluate_potential(0.0))


def main():
    initialize_standard_hamiltonian()


if __name__ == '__main__':
    main()
