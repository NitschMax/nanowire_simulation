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
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def initialize_standard_hamiltonian():
    # define the potential barrier values
    x0 = 0.0
    barrier_height = 5.0
    sigma = 10.0

    # define the hamiltonian parameters, energies in meV, length in m, mass in m_e
    alpha = 5.0
    zeeman = 2.0
    chem_pot = 2.0
    sc_gap = 5.0
    eff_mass = 0.015
    nw_length = 2e+2
    position_grid = 400

    # initialize the potential barriek
    poti = pb.potential_barrier(potential_function, x0, sigma, barrier_height)

    # initialize, build and diagonalize the hamiltonian
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, position_grid, poti)

    hami.build_hamiltonian()
    # Compare diagonalization methods for hami
    # hami.compare_diagonalization_methods()

    # Introduce flags to control zeeman_sweep and phase_sweep
    zeeman_sweep_flag = True
    plot_wavefunctions_flag = True
    phase_sweep_flag = False

    if zeeman_sweep_flag:
        zeeman_sweep(alpha,
                     chem_pot,
                     sc_gap,
                     eff_mass,
                     nw_length,
                     position_grid,
                     poti,
                     num_eigenvalues=100)

    if plot_wavefunctions_flag:
        zeeman = 10.0
        plot_wavefunctions(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                           nw_length, position_grid, poti)

    if phase_sweep_flag:
        phase_sweep(alpha, chem_pot, sc_gap, eff_mass, nw_length,
                    position_grid, poti)


# Define a function that calculates and plots the absolute value of the wavefunctions of the two lowest eigenvalues


def plot_wavefunctions(alpha, zeeman, chem_pot, sc_gap, eff_mass, nw_length,
                       position_grid, poti):
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, position_grid, poti)
    hami.build_hamiltonian()
    hami.diagonalize_hamiltonian()
    eigenvalues, eigenvectors = hami.get_smallest_eigenvalues_and_vectors(2)
    abs_eigenvectors = hami.calculate_abs_wavefunctions(eigenvectors)
    abs_eigenvectors = hami.calculate_abs_gamma_wavefunctions(eigenvectors)

    print('The two lowest eigenvalues are: ', eigenvalues)
    # Include second axis for the potential barrier

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(hami.x, abs_eigenvectors[0], 'r')
    ax1.plot(hami.x, abs_eigenvectors[1], 'g')
    ax2.plot(hami.x, poti(hami.x) / hami.sc_gap, 'b')
    ax1.set_xlabel('Position (nm)')
    ax1.set_ylabel('Wavefunction')
    # Potential barrier is scaled by the superconducting gap
    ax2.set_ylabel(r'Potential barrier ($\Delta$)')
    plt.show()


def phase_sweep(alpha, chem_pot, sc_gap, eff_mass, nw_length, position_grid,
                poti):
    chem_min = -3.0
    chem_max = 8.0
    zeeman_min = 0.0
    zeeman_max = 8.0
    grid_size = 16
    chem_pot_sweep = np.linspace(chem_min, chem_max, grid_size)
    zeeman_sweep = np.linspace(zeeman_min, zeeman_max, grid_size)
    delta_e_sweep = []
    for i, chem_pot in enumerate(chem_pot_sweep):
        delta_e_sweep.append([])
        for j, zeeman in enumerate(zeeman_sweep):
            # print message to keep track of the calculation
            print('Calculating for chem_pot = ', chem_pot, ' and zeeman = ',
                  zeeman)
            print('Progress: ', i * grid_size + j + 1, '/', grid_size**2)
            hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap,
                                           eff_mass, nw_length, position_grid,
                                           poti)
            hami.build_hamiltonian()
            hami.diagonalize_hamiltonian()
            eigenvalues, eigenvectors = hami.get_smallest_eigenvalues_and_vectors(
                2)
            delta_e_sweep[i].append(np.abs(eigenvalues[1] - eigenvalues[0]))

    # Create a 2D plot of the energy difference as a function of the chemical potential and the Zeeman field
    # show colorbar in log scale
    delta_e_sweep = np.array(delta_e_sweep)
    delta_e_sweep = np.log10(delta_e_sweep)
    plt.imshow(delta_e_sweep,
               extent=[zeeman_min, zeeman_max, chem_min, chem_max])

    plt.xlabel('Zeeman field (meV)')
    plt.ylabel('Chemical potential (meV)')
    plt.colorbar()
    plt.show()


def zeeman_sweep(alpha,
                 chem_pot,
                 sc_gap,
                 eff_mass,
                 nw_length,
                 position_grid,
                 poti,
                 num_eigenvalues=10):
    zeeman_sweep = np.linspace(0.0, 10.0, 20)
    eigenvalues_sweep = []
    for i, zeeman in enumerate(zeeman_sweep):
        hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap,
                                       eff_mass, nw_length, position_grid,
                                       poti)
        print('Zeeman field: ', zeeman)
        print('Run', i + 1, 'of', len(zeeman_sweep))
        hami.build_hamiltonian()
        hami.diagonalize_hamiltonian()
        eigenvalues, eigenvectors = hami.get_smallest_eigenvalues_and_vectors(
            num_eigenvalues)
        eigenvalues_sweep.append(eigenvalues)

    # Plot the eigenvalues as a function of zeeman field
    eigenvalues_sweep = np.array(eigenvalues_sweep)
    plt.plot(zeeman_sweep, eigenvalues_sweep, 'k')
    plt.xlabel('Zeeman field (meV)')
    # ylabel energy in units of sc_gap
    plt.ylabel(r'$E [\Delta]$')

    plt.yticks(np.arange(-10, 10, 1.0))
    plt.ylim(-2, 2)

    plt.show()


def main():
    initialize_standard_hamiltonian()


if __name__ == '__main__':
    main()
