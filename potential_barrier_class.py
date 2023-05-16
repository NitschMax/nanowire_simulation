# Define the class of a potential barrier
# It is defined by a function func and two variables x0 and sigma

import unit_conversion as uc


class potential_barrier:

    def __init__(self, func, x0, sigma, barrier_height):
        self.func = func
        self.x0 = x0
        self.sigma = sigma
        self.barrier_height = barrier_height
        self.convert_units()

    def convert_units(self):
        energy_conversion = uc.meV_to_au()
        length_conversion = uc.nm_to_au()
        self.x0 *= length_conversion
        self.sigma *= length_conversion
        self.barrier_height *= energy_conversion

    def __call__(self, x):
        return self.func(x, self.x0, self.sigma) * self.barrier_height

    def __str__(self):
        return 'Potential barrier with x0 = %g and sigma = %g' % (self.x0,
                                                                  self.sigma)

    # Creat a string specifying the directory where the data will be saved
    # Write the floats in format 1.0e-3 as 0.001 with 3 decimals
    # if barrier_height is 0 return 'no_barrier'
    def get_directory_name(self):
        float_format = '.3e'
        dic = {
            'typ': 'gauss',
            'barrier_height': format(self.barrier_height, float_format),
            'sigma': format(self.sigma, float_format),
            'x0': format(self.x0, float_format),
        }
        directory = ''
        for key in dic:
            directory += key + '_' + str(dic[key]) + '_'
        directory = directory[:-1]

        if self.barrier_height == 0:
            directory = 'no_barrier'

        return directory
