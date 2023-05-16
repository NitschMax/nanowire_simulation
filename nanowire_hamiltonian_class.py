import os

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

        self.hamiltonian = None
        self.eigvals = None
        self.eigvecs = None

    def reset_hamiltonian(self):
        self.hamiltonian = None
        self.eigvals = None
        self.eigvecs = None

    def adjust_chem_pot(self, chem_pot):
        self.chem_pot = chem_pot
        self.chem_pot *= uc.meV_to_au()
        self.reset_hamiltonian()

    def adjust_zeeman(self, zeeman):
        self.zeeman = zeeman
        self.zeeman *= uc.meV_to_au()
        self.reset_hamiltonian()

    def get_sc_gap_meV(self):
        return self.sc_gap / uc.meV_to_au()

    # Define a function that return a file name for the Hamiltonian, use a dictionary to create a string
    # Restrict precision to 3 digits, write floats in format with power as in 1.234e-05
    def get_file_name(self):
        name_dict = {
            'alpha': self.alpha,
            'zeeman': self.zeeman,
            'chem_pot': self.chem_pot,
            'sc_gap': self.sc_gap,
            'eff_mass': self.eff_mass,
            'nw_length': self.nw_length,
            'grid_points': self.grid_points
        }
        file_name = ''
        for key, value in name_dict.items():
            file_name += key + '_' + format(value, '.3e') + '_'
        return file_name

    def get_data_directory(self):
        directory = '/Users/ma0274ni/Documents/projects/topological_nanowire/data/' + self.pot_func.get_directory_name(
        ) + '/'
        # Check if the directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory + self.get_file_name()

    def save_data(self):
        if self.eigvals is None or self.eigvecs is None:
            raise ValueError('You need to run the diagonalization first!')
        else:
            np.save(self.get_data_directory() + 'eigvals.npy', self.eigvals)
            np.save(self.get_data_directory() + 'eigvecs.npy', self.eigvecs)

    def check_if_data_exists(self):
        return os.path.exists(self.get_data_directory() + 'eigvals.npy')

    def load_data(self):
        # Check if the data exists
        if not self.check_if_data_exists():
            print('Data does not exist, calculating eigenvalues...')
            self.calculate_only_smallest_eigenvalues()
            self.save_data()
        else:
            self.eigvals = np.load(self.get_data_directory() + 'eigvals.npy')
            self.eigvecs = np.load(self.get_data_directory() + 'eigvecs.npy')
        return self.eigvals, self.eigvecs

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
        sigma_0, sigma_x, sigma_y, sigma_z = self.pauli_matrices()

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
            2 *
            self.eff_mass) * block1 - 1j * self.alpha / self.dx * (+1) * block2
        lower_off_diag_block = -1 / self.dx**2 * 1 / (
            2 *
            self.eff_mass) * block1 - 1j * self.alpha / self.dx * (-1) * block2
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

        return hamiltonian

    def get_smallest_eigenvalues_and_vectors(self, num_eigvals):
        order = np.argsort(np.abs(self.eigvals))
        result_eigvals = self.eigvals[order][:num_eigvals]
        result_eigvecs = self.eigvecs[:, order][:, :num_eigvals]

        re_order = np.argsort(result_eigvals)
        result_eigvals = result_eigvals[re_order]
        result_eigvecs = result_eigvecs[:, re_order]

        return result_eigvals, result_eigvecs

    # Function to calculate the absolute value of the wavefunction on each site
    # This corresponds to the sum of the electron and hole and spin up and spin down wavefunctions on each site
    def adjust_global_phase(self, eigvecs):
        angle = np.angle(eigvecs[0])
        return eigvecs * np.exp(-1j * angle)

    def calculate_operator_expectation(self, eigvec, operator=None):
        # reshape the 1-d array to a 2-d array with first dimension the number of sites and second the spin degrees of freedom with 4 dimensions
        n_sites = eigvec.shape[0] // 4
        reshaped_eigvec = eigvec.reshape((n_sites, 4))

        sigma_0, sigma_x, sigma_y, sigma_z = self.pauli_matrices()
        part_hole_op = np.kron(sigma_y, sigma_y)
        part_hole_op = block_diag(*[part_hole_op for site in self.x])

        if operator is None:
            operator = np.kron(np.eye(2), np.eye(2))
        elif operator == "sigma_z":
            operator = np.kron(np.eye(2), np.array([[1, 0], [0, -1]]))

        abs_wavefunction = np.array(
            [vec.conj().T @ operator @ vec for vec in reshaped_eigvec]).real

        return abs_wavefunction

    # Calculate the absolute value of the psi_0 + i psi_1 wavefunction on each site and
    def calculate_majorana_wavefunctions(self, eigvecs, phi):
        psi_plus = np.exp(1j * phi) * eigvecs[:, 0] + np.exp(
            -1j * phi) * eigvecs[:, 1]
        psi_minus = 1j * np.exp(1j * phi) * eigvecs[:, 0] - 1j * np.exp(
            -1j * phi) * eigvecs[:, 1]
        # Order array of [psi_plus, psi_minus] after the absolute value of the first entry
        result = np.array([psi_plus, psi_minus]).transpose()
        order = np.argsort(np.abs(result[1]))
        return result[:, order]

    def calculate_abs_wavefunctions(self, eigvecs):
        vec_0 = self.calculate_operator_expectation(eigvecs[:, 0])
        vec_1 = self.calculate_operator_expectation(eigvecs[:, 1])

        return vec_0, vec_1

    # Routine that only calculates the eigenvalues with the smallest absolute value via eigsh
    def calculate_only_smallest_eigenvalues(self,
                                            num_eigvals=10,
                                            sigma=0,
                                            positive_first=False):
        eigvals, eigvecs = eigsh(self.hamiltonian,
                                 k=num_eigvals,
                                 which='LM',
                                 mode='normal',
                                 sigma=sigma)

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        if positive_first and num_eigvals == 2:
            return self.return_smallest_positive_and_negative_eigenvalues_and_vectors(
            )
        else:
            return eigvals, eigvecs

    def diagonalize_hamiltonian(self):
        self.eigvals, self.eigvecs = np.linalg.eigh(self.hamiltonian)
        return self.eigvals, self.eigvecs

    def return_smallest_positive_and_negative_eigenvalues_and_vectors(self):
        if self.eigvals is None:
            print("No eigenvalues calculated yet. Calculating now.")
            self.calculate_only_smallest_eigenvalues()

        # Find positive and negative eigenvalues smallest in amplitude and their eigenvectors
        positive_eigval = np.min(self.eigvals[self.eigvals > 0])
        negative_eigval = np.max(self.eigvals[self.eigvals < 0])
        positive_eigvec = self.eigvecs[:, self.eigvals == positive_eigval]
        negative_eigvec = self.eigvecs[:, self.eigvals == negative_eigval]

        eigvals = np.array([positive_eigval, negative_eigval])
        eigvecs = np.concatenate((positive_eigvec, negative_eigvec), axis=1)
        return eigvals, eigvecs

    # Routine that compares differen diagonalization methods
    def compare_diagonalization_methods(self):
        self.build_hamiltonian()

        print(self.diagonalize_hamiltonian()[0])
        print(self.get_smallest_eigenvalues_and_vectors(4)[0])

        eig_arnoldi = self.calculate_only_smallest_eigenvalues(4)[0]
        print(eig_arnoldi)

    def pauli_matrices(self):
        sigma_0 = np.array([[1, 0], [0, 1]])
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        return sigma_0, sigma_x, sigma_y, sigma_z
