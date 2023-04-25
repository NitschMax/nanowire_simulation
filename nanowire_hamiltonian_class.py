# Define the Hamiltonian for a nanowire as a class


class nanowire_hamiltonian:
    # Initialize the Hamiltonian
    def __init__(self, alpha, zeeman, chem_pot, eff_mass, pot_func):
        self.alpha = alpha
        self.zeeman = zeeman
        self.chem_pot = chem_pot
        self.eff_mass = eff_mass
        self.pot_func = pot_func

    def evaluate_potential(self, x):
        return self.pot_func(x)

    # Function that greets the user
    def greet(self):
        print(
            "Hello! I hope you have a nice day! I am a nanowire Hamiltonian.")
