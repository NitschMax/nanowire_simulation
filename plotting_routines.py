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


def separate_majoranas(hami, eigenvectors):
    phi_min_num = optimize.minimize_scalar(
        lambda phi, hami, eigenvectors: -majoranas_avoiding_same_site(
            phi, hami, eigenvectors),
        args=(hami, eigenvectors),
        bounds=(0, np.pi / 2))
    return phi_min_num.x


def overlap_majoranas(hami, eigenvectors):
    phi_max_num = optimize.minimize_scalar(
        lambda phi, hami, eigenvectors: majoranas_avoiding_same_site(
            phi, hami, eigenvectors),
        args=(hami, eigenvectors),
        bounds=(0, np.pi / 2))
    return phi_max_num.x


def plot_wavefunctions(hami,
                       majorana_basis=False,
                       majorana_phi=0.0,
                       minimize_overlap=False,
                       maximize_overlap=False):
    if not hami.check_if_data_exists():
        print('Data does not exist, calculating...')
        hami.build_hamiltonian()
        eigenvalues, eigenvectors = hami.calculate_only_smallest_eigenvalues(
            num_eigvals=2, positive_first=True)
        hami.save_data()
    else:
        hami.load_data()
        eigenvalues, eigenvectors = hami.return_smallest_positive_and_negative_eigenvalues_and_vectors(
        )

    print('The two lowest eigenvalues are: ', eigenvalues)

    # Numerically find phi that minimizes the overlap via optimization
    # Put behind flag to avoid unnecessary calculations
    if minimize_overlap:
        # Try to find minima and maxima of the overlap via numerical optimization
        phi_min_num = separate_majoranas(hami, eigenvectors)

    # Numerically find phi that maximizes the overlap via optimization
    # Put behind flag to avoid unnecessary calculations
    if maximize_overlap:
        phi_max_num = overlap_majoranas(hami, eigenvectors)

    print('The two lowest eigenvalues are: ', eigenvalues)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    plot_wavefunctions_majorana_basis(
        fig,
        ax[0, 0],
        hami,
        eigenvectors,
        phi_min_num,
        majorana_basis,
        title='Minimization of occupation on same site')
    plot_relative_phases_majorana_basis(
        fig,
        ax[1, 0],
        hami,
        eigenvectors,
        phi_min_num,
        majorana_basis,
        title='Relative phase between Majoranas')

    plot_wavefunctions_majorana_basis(
        fig,
        ax[0, 1],
        hami,
        eigenvectors,
        phi_max_num,
        majorana_basis,
        title='Maximization of occupation on same site')
    plot_relative_phases_majorana_basis(
        fig,
        ax[1, 1],
        hami,
        eigenvectors,
        phi_max_num,
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

    # Include second axis for the potential barrier

    ax.plot(hami.x / hami.nw_length, (angles[:, 1] / np.pi))
    ax.set_xlabel('x / L')
    ax.set_ylabel(r'Phase angle / $\pi$')
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


def phase_sweep(hami,
                zeeman_max=4.0,
                chem_min=-3.0,
                chem_max=5.0,
                phase_grid=10,
                log_scale=False,
                mark_transition=False):
    zeeman_min = 1e-3
    chem_pot_sweep = np.linspace(chem_min, chem_max, phase_grid)
    zeeman_sweep = np.linspace(zeeman_min, zeeman_max, phase_grid)

    # Define the Zeeman field andchemical potential sweep as x and y coordinates
    X, Y = np.meshgrid(zeeman_sweep, chem_pot_sweep)
    delta_e_sweep = np.zeros(X.shape)
    eigenvec_sweep = np.zeros(X.shape)

    # Map over 2D grid with np.ndenumerate of Zeeman field values
    for idx, zeeman in np.ndenumerate(X):
        # print message to keep track of the calculation
        chem_pot = Y[idx]

        hami.adjust_zeeman(zeeman)
        hami.adjust_chem_pot(chem_pot)

        print('Progress: ', idx[0] * X.shape[1] + idx[1], '/', X.size)

        if not hami.check_if_data_exists():
            print('Data does not exist, calculating...')
            hami.build_hamiltonian()
            eigenvalues, eigenvecs = hami.calculate_only_smallest_eigenvalues(
                num_eigvals=10)
            hami.save_data()
        else:
            hami.load_data()

        eigenvalues, eigenvecs = hami.return_smallest_positive_and_negative_eigenvalues_and_vectors(
        )
        delta_e_sweep[idx] = np.abs(1 * eigenvalues[0] -
                                    1 * eigenvalues[1]) / hami.sc_gap
        phi = separate_majoranas(hami, eigenvecs)
        phi = overlap_majoranas(hami, eigenvecs)
        eigenvecs = hami.calculate_majorana_wavefunctions(eigenvecs, phi=phi)

        # Reshape the eigenvectors into 2D arrays
        eigenvecs = eigenvecs.reshape(hami.x.shape[0], 4, 2)

        # Calculate tunnel element as sum over sites
        site_idx = 10
        first_tunnel_element = np.sum(eigenvecs[:site_idx, 1, 0], axis=0)
        second_tunnel_element = np.sum(eigenvecs[:site_idx, 1, 1], axis=0)
        data_point = min(np.abs(first_tunnel_element / second_tunnel_element),
                         np.abs(second_tunnel_element / first_tunnel_element))

        # Relative complex angle between the two tunnel elements
        data_point = np.mod(
            np.angle(first_tunnel_element * np.conj(second_tunnel_element)) /
            np.pi + 0.5, 1) - 0.5
        eigenvec_sweep[idx] = np.abs(data_point)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    plot_phase_sweep(hami,
                     fig,
                     axes[0],
                     X,
                     Y,
                     delta_e_sweep,
                     log_scale=log_scale)
    if mark_transition:
        # plot a line in the 2D plot where chem_pot**2 + sc_gap**2 = zeeman**2
        plot_phase_transition(hami, fig, axes[0], chem_min, chem_max,
                              zeeman_min, zeeman_max)

    plot_phase_sweep(hami, fig, axes[1], X, Y, eigenvec_sweep, log_scale=False)
    plot_phase_transition(hami, fig, axes[1], chem_min, chem_max, zeeman_min,
                          zeeman_max)
    fig.tight_layout()
    plt.show()


def plot_phase_sweep(hami, fig, ax, X, Y, delta_e_sweep, log_scale=False):
    # Create a 2D plot with contourf of the energy difference as a function of the chemical potential and the Zeeman field
    # show colorbar and label the axes
    # Show only values until 1e-3
    cmap = plt.get_cmap('viridis')
    if log_scale:
        delta_e_sweep[delta_e_sweep < 1e-3] = 1e-3
        c = ax.contourf(X, Y, np.log10(delta_e_sweep), 100, cmap=cmap)
    else:
        c = ax.contourf(X, Y, delta_e_sweep, 100, cmap=cmap)
    ax.set_xlabel('Zeeman field')
    ax.set_ylabel('Chemical potential')

    fig.colorbar(c)


def plot_phase_transition(hami, fig, ax, chem_min, chem_max, zeeman_min,
                          zeeman_max):
    chem_pot_fine = np.linspace(chem_min, chem_max, 1000)
    ax.plot(np.sqrt(chem_pot_fine**2 + hami.get_sc_gap_meV()**2),
            chem_pot_fine, 'k')
    ax.set_xlim([zeeman_min, zeeman_max])


def zeeman_sweep(hami, zeeman_max=8.0, zeeman_grid=10, num_eigvals=10):
    zeeman_sweep = np.linspace(1e-3, zeeman_max, zeeman_grid)
    eigenvalues_sweep = []
    for i, zeeman in enumerate(zeeman_sweep):
        hami.adjust_zeeman(zeeman)

        print('Zeeman field: ', zeeman)
        print('Run', i + 1, 'of', len(zeeman_sweep))

        if not hami.check_if_data_exists():
            print('Data does not exist, calculating...')
            hami.build_hamiltonian()
            eigenvalues, eigenvecs = hami.calculate_only_smallest_eigenvalues(
                num_eigvals=num_eigvals)
            hami.save_data()
        else:
            eigenvalues, eigenvecs = hami.load_data()

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
