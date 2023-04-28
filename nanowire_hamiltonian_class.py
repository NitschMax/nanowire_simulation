import numpy as np
from scipy.linalg import block_diag
from scipy.sparse.linalg import eigsh

import unit_conversion as uc

# Define the Hamiltonian for a nanowire as a class


class nanowire_hamiltonian:
    # Initialize the Hamiltonian, pot_func is a class that represents the potential
    def __init__(self, alpha, zeeman, chem_pot, sc_gap, eff_mass, nw_length,
                 grid_points, pot_func):
        self.alpha = alpha
        self.zeeman = zeeman
        self.chem_pot = chem_pot
        self.sc_gap = sc_gap
        self.eff_mass = eff_mass
        self.pot_func = pot_func
        self.nw_length = nw_length
        self.grid_points = grid_points

        self.convert_units()
        self.initialize_grid()

    # Conversion of units to atomic units, previously energies were in meV, lengths in nm, masses in m_e
    def convert_units(self):
        energy_conversion = uc.meV_to_au()
        length_conversion = uc.nm_to_au()
        mass_conversion = uc.m_e_to_au()
        self.alpha = self.alpha * energy_conversion * length_conversion
        self.zeeman = self.zeeman * energy_conversion
        self.chem_pot = self.chem_pot * energy_conversion
        self.sc_gap = self.sc_gap * energy_conversion
        self.eff_mass = self.eff_mass * mass_conversion
        self.nw_length = self.nw_length * length_conversion

    # Initialize the grid
    def initialize_grid(self):
        self.dx = self.nw_length / self.grid_points
        self.x = np.arange(0, self.nw_length, self.dx)

    def evaluate_potential(self, x):
        return self.pot_func(x)

    # Function that greets the user
    def greet(self):
        print(
            "Hello! I hope you have a nice day! I am a nanowire Hamiltonian.")

    # Function that builds the matrix representation of the Hamiltonian
    def build_hamiltonian(self):
        # Define the sigma matrices using numpy
        sigma_0 = np.array([[1, 0], [0, 1]])
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        # Build the block matrices representing the repective terms 1. sigma represents electron-hole dof 2. sigma represents spin dof
        block1 = np.kron(sigma_z, sigma_0)
        block2 = np.kron(sigma_z, sigma_y)
        block3 = np.kron(sigma_x, sigma_0)
        block4 = np.kron(sigma_0, sigma_x)

        # Build the diagonal terms
        diag_block_const = (
            +2 / self.dx**2 * 1 / (2 * self.eff_mass) - self.chem_pot
        ) * block1 + self.sc_gap * block3 + self.zeeman * block4
        diag_block = np.array([
            diag_block_const + self.evaluate_potential(site) * block1
            for site in self.x
        ])
        # Cast diag_block to complex
        diag_block = diag_block.astype(complex)

        # Build the off-diagonal terms
        upper_off_diag_block = -1 / self.dx**2 * 1 / (
            2 * self.eff_mass) * block1 - 1j * self.alpha * (+1) * block2
        lower_off_diag_block = -1 / self.dx**2 * 1 / (
            2 * self.eff_mass) * block1 - 1j * self.alpha * (-1) * block2
        upper_off_diag = np.array(
            [upper_off_diag_block for site in self.x[1:]])
        lower_off_diag = np.array(
            [lower_off_diag_block for site in self.x[:-1]])

        # Next we build hamiltonian such that the diagonal conists of diag_block and the off-diagonal consists of upper_off_diag and lower_off_diag
        offset = 4
        aux = np.empty((0, offset), int)

        hamiltonian = block_diag(*diag_block)
        hamiltonian += block_diag(aux, *upper_off_diag, aux.T)
        hamiltonian += block_diag(aux.T, *lower_off_diag, aux)

        self.hamiltonian = hamiltonian

        return hamiltonian / self.sc_gap

    def diagonalize_hamiltonian(self):
        self.eigvals, self.eigvecs = np.linalg.eigh(self.hamiltonian)
        return self.eigvals / self.sc_gap, self.eigvecs

    def get_smallest_eigenvalues_and_vectors(self, num_eigvals):
        order = np.argsort(np.abs(self.eigvals))
        result_eigvals = self.eigvals[order][:num_eigvals]
        result_eigvecs = self.eigvecs[:, order][:, :num_eigvals]

        re_order = np.argsort(result_eigvals)
        result_eigvals = result_eigvals[re_order]
        result_eigvecs = result_eigvecs[:, re_order]

        return result_eigvals / self.sc_gap, result_eigvecs

    # Function to calculate the absolute value of the wavefunction on each site
    # This corresponds to the sum of the electron and hole and spin up and spin down wavefunctions on each site
    def calculate_abs_wavefunctions(self, eigvecs):
        abs_wavefunction = np.abs(eigvecs[0::4])**2 + np.abs(
            eigvecs[1::4])**2 + np.abs(eigvecs[2::4])**2 + np.abs(
                eigvecs[3::4])**2
        return abs_wavefunction

    # Calculate the absolute value of the psi_0 + i psi_1 wavefunction on each site and
    def calculate_abs_gamma_wavefunctions(self, eigvecs):
        psi_plus = eigvecs[:, 0] + eigvecs[:, 1]
        abs_psi_plus = self.calculate_abs_wavefunctions(psi_plus)

        psi_minus = eigvecs[:, 0] - eigvecs[:, 1]
        abs_psi_minus = self.calculate_abs_wavefunctions(psi_minus)

        return abs_psi_plus, abs_psi_minus

    # Routine that only calculates the eigenvalues with the smallest absolute value via eigsh
    def calculate_only_smallest_eigenvalues(self, num_eigvals=10):
        hamiltonian = self.build_hamiltonian()
        eigvals, eigvecs = eigsh(hamiltonian, k=num_eigvals, which='SM')
        return eigvals / self.sc_gap, eigvecs

    # Routine that compares differen diagonalization methods
    def compare_diagonalization_methods(self):
        self.build_hamiltonian()

        print(self.diagonalize_hamiltonian()[0])
        print(self.get_smallest_eigenvalues_and_vectors(4)[0])
        print(self.calculate_only_smallest_eigenvalues(10)[0])
