from scipy.optimize import root
# from atomic_forces.atomOFDFT.python.physics import ThomasFermi, FermiDirac

import numpy as np
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

def Gamma(T, n, Z):
    β = 1/T
    rs = rs_from_n(n)
    return Z**2*β/rs

def Debye_length(T, ni, Zbar):
    ne = Zbar*ni
    EF = Fermi_Energy(ne)
    T_effective = (T**1.8 + (2/3*EF)**1.8)**(1/1.8)
    λD = 1/np.sqrt(  4*π*ne/T_effective ) # Including degeneracy
    # λD = 1/np.sqrt(  4*π*ne/T_effective + 4*π*Zbar**2*ni/T  )  # Including ions
    return λD

def Kappa(T, ni, Zbar):
    rs = rs_from_n(ni)
    λD = Debye_length(T, ni, Zbar)
    return rs/λD

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

# def find_η(Te, ne):
#     """
#     Gets chemical potential in [AU] that gives density ne at temperature Te
#     """
#     f_to_min = lambda η: ThomasFermi.n_TF(Te, η)-ne # η = βμ in ideal gas, no potential case
    
#     root_and_info = root(f_to_min, 0.4,tol=1e-4)

#     η = root_and_info['x'][0]
#     return η

# def P_Ideal_Fermi_Gas(Te, ne):
#     """
#     Gets the noninteracting pressure in AU.
#     """
#     η = find_η(Te, ne)
#     Ithreehalf = FermiDirac.Ithreehalf(η)
#     Θ = Degeneracy_Parameter(Te, ne)
#     P = Te * ne * Θ**(3/2) * Ithreehalf # Goes to 2/5 EF ne
#     return P

# def E_Ideal_Fermi_Gas(Te, ne):
#     """
#     Gets the noninteracting pressure in AU.
#     """
#     η = find_η(Te, ne)
#     Ithreehalf = FermiDirac.Ithreehalf(η)
#     Θ = Degeneracy_Parameter(Te, ne)
#     E = 3/2 * Te * Θ**(3/2) * Ithreehalf # Goes to 3/5 EF
#     return E


def More_TF_Zbar( Z, n_AU, T_AU):
        """
        Finite Temperature Thomas Fermi Charge State using 
        R.M. More, "Pressure Ionization, Resonances, and the
        Continuity of Bound and Free States", Adv. in atomic 
        Mol. Phys., Vol. 21, p. 332 (Table IV).

        Z = atomic number
        n_AU = number density AU
        T = temperature AU
        """

        alpha = 14.3139
        beta = 0.6624
        a1 = 0.003323
        a2 = 0.9718
        a3 = 9.26148e-5
        a4 = 3.10165
        b0 = -1.7630
        b1 = 1.43175
        b2 = 0.31546
        c1 = -0.366667
        c2 = 0.983333

        n_cc = n_AU * AU_to_invcc
        T_eV = T_AU * AU_to_eV

        convert = n_cc*1.6726219e-24
        R = convert/Z
        T0 = T_eV/Z**(4./3.)
        Tf = T0/(1 + T0)
        A = a1*T0**a2 + a3*T0**a4
        B = -np.exp(b0 + b1*Tf + b2*Tf**7)
        C = c1*Tf + c2
        Q1 = A*R**B
        Q = (R**C + Q1**C)**(1/C)
        x = alpha*Q**beta

        return Z*x/(1 + x + np.sqrt(1 + 2.*x))