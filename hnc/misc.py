from scipy.optimize import root
from atomic_forces.atomOFDFT.python.physics import ThomasFermi, FermiDirac

from .constants import *

def Fermi_Energy(ne):
    E_F = 1/(2*m_e) * (3*π**2 * ne)**(2/3)
    return E_F

def Fermi_velocity(ne):
    v_F = np.sqrt(2*Fermi_Energy(ne)/m_e)
    return v_F

def Fermi_wavenumber(ne):
    k_F = Fermi_velocity(ne)*m_e
    return k_F

def Degeneracy_Parameter(Te, ne):
    θ = Te/Fermi_Energy(ne)
    return θ

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

def P_Ideal_Fermi_Gas(Te, ne):
    """
    Gets the noninteracting pressure in AU.
    """
    η = find_η(Te, ne)
    Ithreehalf = FermiDirac.Ithreehalf(η)
    Θ = Degeneracy_Parameter(Te, ne)
    P = Te * ne * Θ**(3/2) * Ithreehalf
    return P

def E_Ideal_Fermi_Gas(Te, ne):
    """
    Gets the noninteracting pressure in AU.
    """
    η = find_η(Te, ne)
    Ithreehalf = FermiDirac.Ithreehalf(η)
    Θ = Degeneracy_Parameter(Te, ne)
    E = 3/2 * Te * Θ**(3/2) * Ithreehalf
    return E


    