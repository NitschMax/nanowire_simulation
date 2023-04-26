# Package to convert units for the simulation
# Energy is in meV, length is in nm, mass is in m_e
# Convert into atomic units


def meV_to_au():
    energy_factor = 1 / 27.2114 / 1000
    return energy_factor


def nm_to_au():
    length_factor = 1 / 0.0529177
    return length_factor


def m_e_to_au():
    mass_factor = 1 / 5.48579909070e-4
    return mass_factor
