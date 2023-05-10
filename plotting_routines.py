# import the nanowire_hamiltonian package
import importlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

import nanowire_hamiltonian_class as nh
import potential_barrier_class as pb

# Get physical values from scipy, hbar and m_e

# Now we want to reload the modules pb and nh
importlib.reload(pb)
importlib.reload(nh)

# reduce the number of decimals for printing in numpy
np.set_printoptions(precision=8)

# Define a function that calculates and plots the absolute value of the wavefunctions of the two lowest eigenvalues


def overlap_between_wavefunctions(eigenvecs, phi=0.0):
    overlap = 4 * (np.exp(2j * phi) * eigenvecs[:, 0] *
                   np.conj(eigenvecs[:, 1])).real
    return overlap


def scalar_overlap_between_wavefunctions(phi, eigenvecs):
    return overlap_between_wavefunctions(eigenvecs, phi=phi).sum()


def inverse_scalar_overlap_between_wavefunctions(phi, eigenvecs):
    return 1 / scalar_overlap_between_wavefunctions(phi, eigenvecs)


def plot_wavefunctions(alpha,
                       zeeman,
                       chem_pot,
                       sc_gap,
                       eff_mass,
                       nw_length,
                       position_grid,
                       poti,
                       majorana_basis=False,
                       majorana_phi=0.0,
                       minimize_overlap=False,
                       maximize_overlap=False):
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, position_grid, poti)
    hami.build_hamiltonian()
    eigenvalues, eigenvectors = hami.calculate_only_smallest_eigenvalues(
        num_eigvals=2)
    # Numerically find phi that minimizes the overlap via optimization
    # Put behind flag to avoid unnecessary calculations
    if minimize_overlap:
        phi_min = optimize.minimize_scalar(
            scalar_overlap_between_wavefunctions,
            args=(eigenvectors, ),
            method='bounded',
            bounds=(0, np.pi / 2))
        print('The phi that minimizes the overlap is: ', phi_min.x / np.pi,
              'pi')
        majorana_phi = phi_min.x

    # Numerically find phi that maximizes the overlap via optimization
    # Put behind flag to avoid unnecessary calculations
    if maximize_overlap:
        phi_max = optimize.minimize_scalar(
            inverse_scalar_overlap_between_wavefunctions,
            args=(eigenvectors, ),
            method='bounded',
            bounds=(0, np.pi / 2))
        print('The phi that maximizes the overlap is: ', phi_max.x / np.pi,
              'pi')
        majorana_phi = phi_max.x

    if majorana_basis:
        eigenvectors = hami.calculate_majorana_wavefunctions(eigenvectors,
                                                             phi=majorana_phi)
    abs_eigenvectors = hami.calculate_abs_wavefunctions(eigenvectors)

    print('The two lowest eigenvalues are: ', eigenvalues)
    # Include second axis for the potential barrier

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(hami.x / hami.nw_length, abs_eigenvectors[0], 'r')
    ax1.plot(hami.x / hami.nw_length, abs_eigenvectors[1], 'g')
    ax2.plot(hami.x / hami.nw_length, poti(hami.x) / hami.sc_gap, 'b')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Wavefunction')
    # Potential barrier is scaled by the superconducting gap
    ax2.set_ylabel(r'Potential barrier ($\Delta$)')
    plt.show()


def phase_sweep(alpha,
                chem_pot,
                sc_gap,
                eff_mass,
                nw_length,
                position_grid,
                poti,
                zeeman_max=4.0,
                chem_min=-3.0,
                chem_max=5.0,
                phase_grid=10):
    zeeman_min = 1e-3
    chem_pot_sweep = np.linspace(chem_min, chem_max, phase_grid)
    zeeman_sweep = np.linspace(zeeman_min, zeeman_max, phase_grid)

    # Define the Zeeman field andchemical potential sweep as x and y coordinates
    X, Y = np.meshgrid(zeeman_sweep, chem_pot_sweep)
    delta_e_sweep = np.zeros(X.shape)

    # Map over 2D grid with np.ndenumerate of Zeeman field values
    for idx, zeeman in np.ndenumerate(X):
        # print message to keep track of the calculation
        chem_pot = Y[idx]

        hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap,
                                       eff_mass, nw_length, position_grid,
                                       poti)
        print('Calculating for chem_pot = ', chem_pot, ' and zeeman = ',
              zeeman)
        print('Progress: ', idx[0] * X.shape[1] + idx[1], '/', X.size)
        hami.build_hamiltonian()
        eigenvalues, eigenvecs = hami.calculate_only_smallest_eigenvalues(
            num_eigvals=10)
        delta_e_sweep[idx] = np.abs(0 * eigenvalues[1] -
                                    2 * eigenvalues[0]) / hami.sc_gap

    # Create a 2D plot with contourf of the energy difference as a function of the chemical potential and the Zeeman field
    # show colorbar in log scale down to 1e-3
    delta_e_sweep = np.array(delta_e_sweep)
    delta_e_sweep = np.log10(delta_e_sweep)
    delta_e_sweep[delta_e_sweep < -3] = -3
    plt.contourf(X, Y, delta_e_sweep, 100, levels=np.linspace(-3, 1, 100))

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
                 zeeman_max=8.0,
                 zeeman_grid=10,
                 num_eigvals=10):
    zeeman_sweep = np.linspace(1e-3, zeeman_max, zeeman_grid)
    eigenvalues_sweep = []
    for i, zeeman in enumerate(zeeman_sweep):
        hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap,
                                       eff_mass, nw_length, position_grid,
                                       poti)
        print('Zeeman field: ', zeeman)
        print('Run', i + 1, 'of', len(zeeman_sweep))
        hami.build_hamiltonian()
        eigenvalues, eigenvecs = hami.calculate_only_smallest_eigenvalues(
            num_eigvals=num_eigvals)
        # Use function sort_eigenvalues to sort the eigenvalues and eigenvectors
        eigenvalues, eigenvecs = sort_eigenvalues(eigenvalues, eigenvecs)
        eigenvalues_sweep.append(eigenvalues)

    # Plot the eigenvalues as a function of zeeman field
    eigenvalues_sweep = np.array(eigenvalues_sweep)
    plt.plot(zeeman_sweep, eigenvalues_sweep / hami.sc_gap, 'k')
    plt.xlabel('Zeeman field (meV)')
    # ylabel energy in units of sc_gap
    plt.ylabel(r'$E [\Delta]$')

    # plt.yticks(np.arange(-10, 10, 1.0))
    # plt.ylim(-2, 2)

    plt.show()


def sort_eigenvalues(eigvals, eigvecs):
    order = np.argsort(np.abs(eigvals))
    result_eigvals = eigvals[order]
    result_eigvecs = eigvecs[:, order]

    re_order = np.argsort(result_eigvals)
    result_eigvals = result_eigvals[re_order]
    result_eigvecs = result_eigvecs[:, re_order]

    return result_eigvals, result_eigvecs
