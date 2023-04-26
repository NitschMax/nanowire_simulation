import numpy as np

import unit_conversion as uc

# Define the Hamiltonian for a nanowire as a class


class nanowire_hamiltonian:
    # Initialize the Hamiltonian
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
    def build_hamiltonian(self, x):
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

        # Build the terms using the previously defined blocks
        diag_block = (-2 / self.dx**2 * 1 / (2 * self.eff_mass) -
                      self.chem_pot + self.evaluate_potential(x)
                      ) * block1 + self.sc_gap * block3 + self.zeeman * block4
        print("Diagonal block 1: \n", diag_block)
