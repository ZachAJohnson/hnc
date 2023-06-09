import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

from math import isnan

from pandas import read_csv

from scipy.special import erfc

# Build Components for QSP's
from constants import *

class QSP_HNC():


    def __init__(self, Z, A, Zstar, Te, Ti, ri, ne, which_Tij='thermal', r_c = 3/5, verbose= False):
        self.Z = Z
        self.A = A
        self.Zstar = Zstar
        self.Te = Te
        self.Ti = Ti
        self.ri = ri
        self.ne = ne
        self.r_c = r_c
        self.which_Tij = which_Tij 

        self.initialize_physics(Z, A, Zstar, Te, Ti, ri, ne, verbose=verbose)

    def initialize_physics(self, Z, A, Zstar, Te, Ti, ri, ne, verbose=False):
        # [AU]

        self.ri = ri
        self.ni = self.n_from_rs(ri)
        self.m_i = m_p*self.A
        
        # Actual Temperatures and inverse temperatures
        self.βi  = 1/self.Ti
        self.βe  = 1/self.Te
        if self.which_Tij=='thermal':
            self.Tie = self.Tei_thermal(Te, Ti)
        elif self.which_Tij=='geometric':
            self.Tie = self.Tei_geometric(Te, Ti)
        self.βie = 1/self.Tie

        self.ne = ne#self.Zstar*self.ni
        self.re = self.rs_from_n(self.ne)

        self.E_F = 1/(2*m_e) * (3*π**2 * self.ne)**(2/3)
        self.v_F = np.sqrt(2*self.E_F/m_e)
        self.k_F = self.v_F*m_e
        self.θ   = self.Te/self.E_F

        #Construct effective electron temperatures. https://journals-aps-org.proxy.lib.umich.edu/prl/pdf/10.1103/PhysRevLett.84.959
        self.Tq  = self.E_F/(1.594 - 0.3160*np.sqrt(self.re) + 0.0240*self.re) #DMC
        # self.Tq  = self.E_F/(1.3251 - 0.1779*np.sqrt(self.re) + 0.0*self.re) #VMC
        self.Te_c  = self.make_Te(self.Te, self.Tq)

        self.lambda_TF = np.sqrt( self.Te / (4*π*self.ne)  )

        if self.which_Tij=='thermal':
            self.Tie_c = self.Tei_thermal(self.Te_c, self.Tq)
        elif self.which_Tij=='geometric':
            self.Tie_c = self.Tei_geometric(self.Te_c, self.Tq)

        self.βe_c  = 1/self.Te_c
        self.βie_c = 1/self.Tie_c 
        self.Tij = np.array([[self.Ti, self.Tie_c],
                              [self.Tie_c, self.Te_c]])
        self.μ = self.m_i*m_e/(m_e+self.m_i)
        self.mij = np.array([[self.m_i, self.μ],
                              [self.μ, m_e]])

        self.Λee  = 1/np.sqrt(π*m_e*self.Te_c )/self.ri
        self.Λei  = 1/np.sqrt(2*π*m_e*self.Tie_c )/self.ri 
        
        self.Γee =  self.βe_c/self.ri 
        self.Γei = -self.Zstar*self.βie_c/self.ri
        self.Γii =  self.Zstar**2*self.βi/self.ri 

        self.Γ_matrix = np.array(  [[self.Γii,  self.Γei],
                                    [self.Γei,  self.Γee]])
        
        if verbose:
            print("Λei = {0:.3f}".format(self.Λei))
            print("Λee = {0:.3f}".format(self.Λee))
            
            print("Γii={0:.3f}, Γie={1:.3f}, Γee={2:.3f} ".format(self.Γii, self.Γei, self.Γee))
            print("r_i={0:.3f}".format(self.ri))
            print("r_e={0:.3f}".format(self.re))
            print("r_c={0:.3f}".format(self.r_c))
            print("θ  ={0:.2e}".format(self.θ))

    @staticmethod
    def make_Te(Te, Tq):
        Te_c  = np.sqrt(Tq**2 + Te**2)
        return Te_c

    @staticmethod
    def n_from_rs( rs):
        return 1/(4/3*π*rs**3)

    @staticmethod
    def rs_from_n(n):
        return (4/3*π*n)**(-1/3)

    def get_κ(self):
        kTF = np.sqrt(  4*π*self.ne  /self.Te_c  )
        return kTF*self.ri

    def Tei_geometric(self, Te, Ti):
        return np.sqrt(Te*Ti)

    def Tei_thermal(self, Te, Ti):
        μ_ei = self.m_i*m_e/(m_e+self.m_i) #reduced mass
        Tie = μ_ei * (Te/m_e + Ti/self.m_i) #thermal velocity defined Tei
        return Tie

    def βv_Deutsch(self, Γ, r, Λ):
        # return Γ/r* ( 1 -  np.exp(-np.sqrt(2)π*r/Λe) ) # for Sarkas Λ
        return Γ/r* ( 1 -  np.exp(-r/Λ) )

    # Kelbg looks really weird??
    def βv_Kelbg(self, Γ,r, Λ):
        return Γ/r*( 1 - np.exp(-r**2/(Λ**2*2*π)) + r/Λ*erfc(r/Λ/np.sqrt(2*π)) )
        # return Γ/r*( 1 - np.exp(-2*π*r**2/Λe**2)+ np.sqrt(2)*π*r/Λe*erfc(np.sqrt(2*π)*r/Λe))

    def βv_Pauli(self, r, Λ):
        #Sarkas
        # return  np.log(2) * np.exp(-4*π* r**2 /( Λe**2))
        #return -np.log(1 - 0.5*np.exp(-r**2/(2*π*Λe**2)) )
        return np.log(2)*np.exp(-r**2/(π*np.log(2)*Λ**2))

    
    ######### Build Actual QSP's

    def βvee(self, r):
        return self.βv_Deutsch(self.Γee,r, self.Λee) + self.βv_Pauli(r,self.Λee)

    def βvei(self, r):
        return self.βv_Deutsch(self.Γei,r,self.Λei) #+ (Z-Zstar)*βv_Pauli(r)

    def βvei_atomic(self, r, core_height=0):
        r_c  = self.r_c #3/5 r_s in linear n_b(r) model
        Λ, Γ = self.Λei, self.Γei
        one_over_r_cutoff = np.heaviside(r - r_c, 0.5)/r + core_height/r_c * np.heaviside(r_c - r,0.5)
        βvei_pseudopotential = Γ*one_over_r_cutoff
        return βvei_pseudopotential * ( 1 -  np.exp(-r/Λ) )

    # def βvei_atomic(self, r, core_height=1):
    #     r_c  = self.r_c #3/5 r_s in linear n_b(r) model
    #     Λ, Γ = self.Λei, self.Γei
    #     inside_core = np.heaviside(r_c - r,0.5)
    #     outside_core = np.heaviside(r - r_c, 0.5)
    #     one_over_r_cutoff = inside_core*core_height  + outside_core*r_c/r  
    #     βvei_pseudopotential = one_over_r_cutoff
    #     return -2*βvei_pseudopotential * ( 1 -  np.exp(-r/Λ) )

    # def βvei_atomic(self, r, core_height=1):
    #     r_c  = self.r_c #3/5 r_s in linear n_b(r) model
    #     Λ, Γ = self.Λei, self.Γei
    #     Λ_atomic = np.sqrt(r_c**2 + Λ**2)
        
    #     unnormalized_βvei = 1/r * ( 1 -  np.exp(-r/Λ_atomic) )
    #     unnormalized_βvei_at_0 = 1/Λ_atomic
    #     correction_factor_from_h_c = 2
    #     return correction_factor_from_h_c *(-unnormalized_βvei  * 1/unnormalized_βvei_at_0)

    def βvii(self, r):
        return self.Γii/r 

    def βv_Yukawa(self, r):
                return self.Γii/r * np.exp(-r*self.ri/self.lambda_TF)
        