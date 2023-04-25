# Define the class of a potential barrier
# It is defined by a function func and two variables x0 and sigma


class potential_barrier:

    def __init__(self, func, x0, sigma):
        self.func = func
        self.x0 = x0
        self.sigma = sigma

    def __call__(self, x):
        return self.func(x, self.x0, self.sigma)

    def __str__(self):
        return 'Potential barrier with x0 = %g and sigma = %g' % (self.x0,
                                                                  self.sigma)
