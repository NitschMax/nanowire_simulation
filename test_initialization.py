# import the nanowire_hamiltonian package
import nanowire_hamiltonian as nh


def initialize_standard_hamiltonian():
    alpha = 1.0
    zeeman = 1.0
    chem_pot = 1.0
    eff_mass = 1.0
    hami = nh.nanowire_hamiltonian(alpha, zeeman, chem_pot, eff_mass)
    hami.greet()


def main():
    initialize_standard_hamiltonian()


if __name__ == '__main__':
    main()
