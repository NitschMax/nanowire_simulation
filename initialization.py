# import the nanowire_hamiltonian package
import importlib

import matplotlib.pyplot as plt
import numpy as np

import nanowire_hamiltonian_class as nh
import potential_barrier_class as pb

# Get physical values from scipy, hbar and m_e

# Now we want to reload the modules pb and nh
importlib.reload(pb)
importlib.reload(nh)

# reduce the number of decimals for printing in numpy
np.set_printoptions(precision=8)


def potential_function(x, x0, sigma):
    return np.exp((x - x0)**2 / (2 * sigma**2))


def initialize_standard_hamiltonian():
    # define the potential barrier values
    x0 = 0.0
    barrier_height = 10.0
    sigma = 100.0

    # define the hamiltonian parameters, energies in meV, length in m, mass in m_e
    alpha = 50.0
    zeeman = 2.0
    chem_pot = 5.0
    sc_gap = 0.5
    eff_mass = 0.015
    nw_length = 4e+3
    position_grid = 100

    # initialize the potential barriek
    poti = pb.potential_barrier(potential_function, x0, sigma, barrier_height)

    # initialize, build and diagonalize the hamiltonian
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, position_grid, poti)

    # Do a sweep over zeeman field, initialize the hamiltonian and diagonalize for each value
    # do_zeeman_sweep(alpha, chem_pot, sc_gap, eff_mass, nw_length,
    #                 position_grid, poti)

    # Only compute two eigenvalues closest to zero
    # use argsort to sort the eigenvalues for their absolute value
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, position_grid, poti)
    hamiltonian_matrix = hami.build_hamiltonian()
    eigvals = np.linalg.eigvalsh(hamiltonian_matrix)
    ordering = np.abs(eigvals).argsort()
    eigvals = eigvals[ordering]

    print("The two lowest eigenvalues are:", eigvals[0], "and", eigvals[1])
    delta_e = eigvals[1] - eigvals[0]
    print("The energy difference between the two lowest eigenvalues is:",
          delta_e)

    # Perform a 2D sweep over the chemical potential and the Zeeman field
    # initialize the hamiltonian and diagonalize for each value
    # use argsort to sort the eigenvalues for their absolute value
    # plot the absolute difference between the two lowest eigenvalues as a function of the chemical potential and the Zeeman field
    # perform calculation and sweep in an external function called do_2d_sweep
    do_2d_sweep(alpha, chem_pot, sc_gap, eff_mass, nw_length, position_grid,
                poti)


def do_2d_sweep(alpha, chem_pot, sc_gap, eff_mass, nw_length, position_grid,
                poti):
    zeeman_sweep = np.linspace(0.0, 20.0, 10)
    chem_pot_sweep = np.linspace(0.0, 20.0, 10)
    delta_e_sweep = []
    for i, zeeman in enumerate(zeeman_sweep):
        delta_e_sweep.append([])
        for j, chem_pot in enumerate(chem_pot_sweep):
            hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap,
                                           eff_mass, nw_length, position_grid,
                                           poti)
            hamiltonian_matrix = hami.build_hamiltonian()
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
            ordering = np.abs(eigenvalues).argsort()
            eigenvalues = eigenvalues[ordering]
            delta_e_sweep[i].append(np.abs(eigenvalues[1] - eigenvalues[0]))

    # Create a 2D plot of the energy difference as a function of the chemical potential and the Zeeman field
    delta_e_sweep = np.array(delta_e_sweep)
    plt.imshow(delta_e_sweep, extent=[0.0, 20.0, 0.0, 20.0])
    plt.xlabel('Chemical potential (meV)')
    plt.ylabel('Zeeman field (meV)')
    plt.colorbar()
    plt.show()


def do_zeeman_sweep(alpha, chem_pot, sc_gap, eff_mass, nw_length,
                    position_grid, poti):
    zeeman_sweep = np.linspace(0.0, 20.0, 10)
    eigenvalues_sweep = []
    for i, zeeman in enumerate(zeeman_sweep):
        hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap,
                                       eff_mass, nw_length, position_grid,
                                       poti)
        hamiltonian_matrix = hami.build_hamiltonian()
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
        eigenvalues_sweep.append(eigenvalues)

    # Plot the eigenvalues as a function of zeeman field
    eigenvalues_sweep = np.array(eigenvalues_sweep)
    plt.plot(zeeman_sweep, eigenvalues_sweep)
    plt.xlabel('Zeeman field (meV)')
    plt.ylabel('Eigenvalues (meV)')
    plt.ylim(-100.0, 100.0)
    plt.show()


def main():
    initialize_standard_hamiltonian()


if __name__ == '__main__':
    main()
