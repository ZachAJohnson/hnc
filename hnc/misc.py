from scipy.optimize import root
from atomic_forces.atomOFDFT.python.physics import ThomasFermi

from .constants import *

def n_from_rs( rs):
    """
    Sphere radius to density, in any units
    """
    return 1/(4/3*π*rs**3)

def rs_from_n(n):
    """
    Density to sphere radius, in any units.
    """
    return (4/3*π*n)**(-1/3)

def find_η(Te, ne):
    """
    Gets chemical potential in [AU] that gives density ne at temperature Te
    """
    f_to_min = lambda η: ThomasFermi.n_TF(Te, η)-ne # η = βμ in ideal gas, no potential case
    
    root_and_info = root(f_to_min, 0.4,tol=1e-4)

    η = root_and_info['x'][0]
    return η
