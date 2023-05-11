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


def overlaps_majoranas_analytical(eigenvecs, phi=0.0, max_idx=-1):
    overlap = (np.exp(+2j * phi) * eigenvecs[:max_idx, 0] *
               np.conj(eigenvecs[:max_idx, 1])).real
    return overlap


def scalar_overlaps_majoranas_analytical(phi, eigenvecs, max_idx=-1):
    return np.abs(1 - overlaps_majoranas_analytical(
        eigenvecs, phi=phi, max_idx=max_idx).sum())**2


def scalar_majoranas_avoiding_same_site(eigenvecs):
    return np.abs(np.abs(eigenvecs[:, 0])**2 -
                  np.abs(eigenvecs[:, 1])**2).sum()
    return np.abs((eigenvecs[:, 0] * np.conj(eigenvecs[:, 1])).sum())**2


def majoranas_avoiding_same_site(phi, hami, eigenvectors):
    return scalar_majoranas_avoiding_same_site(
        hami.calculate_majorana_wavefunctions(eigenvectors, phi))


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
        num_eigvals=2, positive_first=True)

    # Numerically find phi that minimizes the overlap via optimization
    # Put behind flag to avoid unnecessary calculations
    if minimize_overlap:
        # Try to find minima and maxima of the overlap via numerical optimization
        phi_min_num = optimize.minimize_scalar(
            lambda phi, hami, eigenvectors: majoranas_avoiding_same_site(
                phi, hami, eigenvectors),
            args=(hami, eigenvectors),
            bounds=(0, np.pi / 2))
        print('The phi that minimizes the Majorana avoidance is: ',
              phi_min_num.x / np.pi, 'pi')
        print('The nummerically minimized avoidance is: ',
              majoranas_avoiding_same_site(phi_min_num.x, hami, eigenvectors))

    # Numerically find phi that maximizes the overlap via optimization
    # First guess is pi/4
    # Put behind flag to avoid unnecessary calculations
    if maximize_overlap:
        phi_max_num = optimize.minimize_scalar(
            lambda phi, hami, eigenvectors: -majoranas_avoiding_same_site(
                phi, hami, eigenvectors),
            args=(hami, eigenvectors),
            bounds=(0, np.pi / 2))
        print('The phi that maximizes the Majorana avoidance is: ',
              phi_max_num.x / np.pi, 'pi')
        print('The nummerically maximized avoidance is: ',
              majoranas_avoiding_same_site(phi_max_num.x, hami, eigenvectors))

    print('The two lowest eigenvalues are: ', eigenvalues)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    plot_wavefunctions_majorana_basis(
        fig,
        ax[0, 0],
        hami,
        eigenvectors,
        phi_min_num.x,
        majorana_basis,
        title='Maximization of occupation on same site')
    plot_relative_phases_majorana_basis(
        fig,
        ax[1, 0],
        hami,
        eigenvectors,
        phi_min_num.x,
        majorana_basis,
        title='Relative phase between Majoranas')

    plot_wavefunctions_majorana_basis(
        fig,
        ax[0, 1],
        hami,
        eigenvectors,
        phi_max_num.x,
        majorana_basis,
        title='Minimization of occupation on same site')
    plot_relative_phases_majorana_basis(
        fig,
        ax[1, 1],
        hami,
        eigenvectors,
        phi_max_num.x,
        majorana_basis,
        title='Relative phase between Majoranas')
    fig.tight_layout()
    plt.show()


def plot_relative_phases_majorana_basis(fig,
                                        ax,
                                        hami,
                                        eigenvectors,
                                        majorana_phi,
                                        majorana_basis,
                                        title=''):
    if majorana_basis:
        eigenvectors = hami.calculate_majorana_wavefunctions(eigenvectors,
                                                             phi=majorana_phi)

    angles = np.angle(eigenvectors[:, 0] *
                      np.conj(eigenvectors[:, 1])).reshape(hami.x.shape[0], 4)
    print('The phase difference between the two Majoranas is: ',
          angles / np.pi, 'pi')

    # Include second axis for the potential barrier

    ax.plot(hami.x / hami.nw_length, (angles[:, 1] / np.pi))
    ax.set_xlabel('x / L')
    ax.set_ylabel('Phase angle / $\pi$')
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_title(title)


# Define a function that plots the wavefunction in the Majorana basis
def plot_wavefunctions_majorana_basis(fig,
                                      ax,
                                      hami,
                                      eigenvectors,
                                      majorana_phi,
                                      majorana_basis,
                                      title=''):
    if majorana_basis:
        eigenvectors = hami.calculate_majorana_wavefunctions(eigenvectors,
                                                             phi=majorana_phi)
    print('Absolute value of the wavefunctions: ',
          np.abs(eigenvectors[:, 0].reshape(hami.x.shape[0], 4)))
    abs_eigenvectors = hami.calculate_abs_wavefunctions(eigenvectors)

    # Include second axis for the potential barrier

    ax2 = ax.twinx()
    ax.plot(hami.x / hami.nw_length, abs_eigenvectors[0], 'r')
    ax.plot(hami.x / hami.nw_length, abs_eigenvectors[1], 'g')
    ax2.plot(hami.x / hami.nw_length, hami.pot_func(hami.x) / hami.sc_gap, 'b')
    ax.set_xlabel('Position')
    ax.set_ylabel('Wavefunction')
    # Potential barrier is scaled by the superconducting gap
    ax2.set_ylabel(r'Potential barrier ($\Delta$)')
    ax.set_title(title)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))


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
