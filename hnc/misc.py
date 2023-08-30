
from .constants import *

@staticmethod
def n_from_rs( rs):
    """
    Sphere radius to density, in any units
    """
    return 1/(4/3*π*rs**3)

@staticmethod
def rs_from_n(n):
    """
    Density to sphere radius, in any units.
    """
    return (4/3*π*n)**(-1/3)
