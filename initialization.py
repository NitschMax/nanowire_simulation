# import the nanowire_hamiltonian package
import importlib

import numpy as np

import nanowire_hamiltonian_class as nh
import plotting_routines as pr
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
    barrier_height = 10.0
    sigma = 200.0

    # Zeeeman field sweep parameters
    zeeman_max = 4.0
    zeeman_grid = 10
    num_eigvals = 20

    # Phase sweep parameters
    chem_min = -1.0
    chem_max = 7.5
    phase_grid = 30

    # Wavefunction plot parameters
    majorana_basis = True
    majorana_phi = 0.0 * np.pi
    minimize_overlap = True
    maximize_overlap = True

    # define the hamiltonian parameters, energies in meV, length in nm, mass in m_e
    alpha = 50.0
    zeeman = 4.0
    chem_pot = +6.0
    sc_gap = 0.5
    eff_mass = 0.015
    nw_length = 2e+3
    position_grid = 500

    # initialize the potential barriek
    poti = pb.potential_barrier(potential_function, x0, sigma, barrier_height)

    # initialize, build and diagonalize the hamiltonian
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, sc_gap, eff_mass,
                                   nw_length, position_grid, poti)

    hami.build_hamiltonian()
    # Compare diagonalization methods for hami
    # hami.compare_diagonalization_methods()

    # Introduce flags to control zeeman_sweep, plot_wavefunctions and phase_sweep
    zeeman_sweep_flag = False
    plot_wavefunctions_flag = True
    phase_sweep_flag = False

    if zeeman_sweep_flag:
        pr.zeeman_sweep(alpha,
                        chem_pot,
                        sc_gap,
                        eff_mass,
                        nw_length,
                        position_grid,
                        poti,
                        zeeman_max=zeeman_max,
                        zeeman_grid=zeeman_grid,
                        num_eigvals=num_eigvals)

    if plot_wavefunctions_flag:
        zeeman = zeeman_max
        pr.plot_wavefunctions(alpha,
                              zeeman,
                              chem_pot,
                              sc_gap,
                              eff_mass,
                              nw_length,
                              position_grid,
                              poti,
                              majorana_basis=majorana_basis,
                              majorana_phi=majorana_phi,
                              minimize_overlap=minimize_overlap,
                              maximize_overlap=maximize_overlap)

    if phase_sweep_flag:
        pr.phase_sweep(alpha,
                       chem_pot,
                       sc_gap,
                       eff_mass,
                       nw_length,
                       position_grid,
                       poti,
                       zeeman_max=zeeman_max,
                       chem_min=chem_min,
                       chem_max=chem_max,
                       phase_grid=phase_grid)
    return


def main():
    initialize_standard_hamiltonian()


if __name__ == '__main__':
    main()
