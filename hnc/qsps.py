import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

from math import isnan

from pandas import read_csv

from scipy.special import erfc

# Build Components for QSP's
from .constants import *
from .misc import rs_from_n, n_from_rs, P_Ideal_Fermi_Gas

class Quantum_Statistical_Potentials():
    """
    Class defining classical mapping potentials for electron-ion systems where the electrons are partially degenerate.  
    """

    def __init__(self, Z, A, Zbar, Te, Ti, ri, ne, which_Tij='thermal', r_c = 3/5, verbose= False, Te_c_type = 'Fermi', custom_Te_c = None):
        self.Z = Z
        self.A = A
        self.Zbar = Zbar
        self.Te = Te
        self.Ti = Ti
        self.ri = ri
        self.ne = ne
        self.r_c = r_c
        self.which_Tij = which_Tij 
        self.Te_c_type = Te_c_type
        self.custom_Te_c = custom_Te_c

        self.initialize_physics(Z, A, Zbar, Te, Ti, ri, ne, verbose=verbose)

    def initialize_physics(self, Z, A, Zbar, Te, Ti, ri, ne, verbose=False):
        # [AU]

        self.ri = ri
        self.ni = n_from_rs(ri)
        self.m_i = m_p*self.A
        
        self.ne = ne#self.Zbar*self.ni
        self.re = rs_from_n(self.ne)

        self.E_F = 1/(2*m_e) * (3*π**2 * self.ne)**(2/3)
        self.v_F = np.sqrt(2*self.E_F/m_e)
        self.k_F = self.v_F*m_e
        self.θ   = self.Te/self.E_F

        #Construct effective electron temperatures. https://journals-aps-org.proxy.lib.umich.edu/prl/pdf/10.1103/PhysRevLett.84.959
        if self.Te_c_type=='DMC':
            self.Tq  = self.E_F/(1.594 - 0.3160*np.sqrt(self.re) + 0.0240*self.re) #DMC
            self.Te_c  = self.make_Te(self.Te, self.Tq)
        elif self.Te_c_type=='VMC':
            self.Tq  = self.E_F/(1.3251 - 0.1779*np.sqrt(self.re) + 0.0*self.re) #VMC
            self.Te_c  = self.make_Te(self.Te, self.Tq)
        elif self.Te_c_type=='Fermi':
            self.Te_c = P_Ideal_Fermi_Gas(self.Te, self.ne)/self.ne # Matches free Fermi gas pressure exactly    
        elif self.Te_c_type=='classical':   
            self.Te_c = self.Te
        elif self.Te_c_type=='custom':
            self.Te_c = self.custom_Te_c

        self.lambda_TF = np.sqrt( self.Te / (4*π*self.ne)  )

        if self.which_Tij=='thermal':
            self.Tie = self.Tei_thermal(Te, Ti)
            self.Tie_c = self.Tei_thermal(self.Te_c, self.Ti)
        elif self.which_Tij=='geometric':
            self.Tie = self.Tei_geometric(Te, Ti)
            self.Tie_c = self.Tei_geometric(self.Te_c, self.Ti)

        # Inverse temperatures
        self.βi  = 1/self.Ti
        self.βe  = 1/self.Te

        self.βie = 1/self.Tie
        self.βe_c  = 1/self.Te_c
        self.βie_c = 1/self.Tie_c 

        self.Tij = np.array([[self.Ti, self.Tie_c],
                              [self.Tie_c, self.Te_c]])
        self.μ = self.m_i*m_e/(m_e+self.m_i)
        self.mij = np.array([[self.m_i, 2*self.μ],
                              [2*self.μ, m_e]])

        self.Λee  = 1/np.sqrt(2*π*(m_e/2)*self.Te_c )/self.ri 
        self.Λei  = 1/np.sqrt(2*π*self.μ*self.Tie_c )/self.ri

        # self.Λee  = 1/np.sqrt(2*(m_e/2)*self.Te_c )/self.ri #Bredow 
        # self.Λei  = 1/np.sqrt(2*self.μ*self.Tie_c )/self.ri #Bredow         
        
        self.Γee =  self.βe_c/self.ri 
        self.Γei = -self.Zbar*self.βie_c/self.ri
        self.Γii =  self.Zbar**2*self.βi/self.ri 

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

    def get_κ(self):
        kTF = np.sqrt(  4*π*self.ne**2  /P_Ideal_Fermi_Gas(self.Te_c, self.ne)  )
        return kTF*self.ri

    def Tei_geometric(self, Te, Ti): # Returns geometric mean of two temperatures
        return np.sqrt(Te*Ti)

    def Tei_thermal(self, Te, Ti): # Averages thermal velocities. If Ti=0 or mi->infinity, Tei_thermal = Te
        μ_ei = self.m_i*m_e/(m_e+self.m_i) #reduced mass
        Tie = μ_ei * (Te/m_e + Ti/self.m_i) #thermal velocity defined Tei
        return Tie

    # Actual QSP's 
    def βv_Deutsch(self, Γ, r, Λ): # Checked exactly for Λ defined by Λ_ij = hbar/sqrt(  2 π T_ij μ_ij )
        return Γ/r* ( 1 -  np.exp(-r/Λ) )   

    def βv_Kelbg(self, Γ, r, Λ): # Checked exactly for Λ defined by Λ_ij = hbar/sqrt(  2 π T_ij μ_ij )
        return Γ/r*( 1 - np.exp(-r**2/(π*Λ**2)) + r/Λ * erfc( r/(Λ*np.sqrt(π)) )  )

    def βv_Improved_Kelbg(self, Γ, r, Λ, γ=1): # Checked exactly for Λ defined by Λ_ij = hbar/sqrt(  2 π T_ij μ_ij )
        return Γ/r*( 1 - np.exp(-r**2/(π*Λ**2)) + r/(Λ*γ) * erfc( γ*r/(Λ*np.sqrt(π)) )  )

    def βv_Pauli(self, r, type='Lado'):
        #Sarkas
        if type=='ideal':# Adapted eq. 15 of Jones & Murillo from Uhlenbeck and Gropper (UG
            return -np.log(1 - 0.5*np.exp(-r**2/(π*self.Λee**2)) )         

        if type=='Lado': # Adapted eq. 44/45 of Jones & Murillo from Lado
            τ = self.Te/self.E_F
            a1, a2, a3 = 0.2975, 6.090, 1.541
            b1, b2, b3, b4 = 0.0842, 0.1027, 1.096, 1.359
            A = 1 + a1/(1+a2*τ**a3)
            B = 1 + b1*np.exp(-b2*τ**b3)/τ**b4
            return -np.log(1-0.5*A*np.exp(-B*r**2/(π*self.Λee**2))) 
       
    ######### Build Actual QSP's
    def βvee(self, r):
        return self.βv_Deutsch(self.Γee,r, self.Λee) + self.βv_Pauli(r)

    def βvei(self, r):
        """
        Using fit from Filinov et al 2004 PHYSICAL REVIEW E 70, 046411 (2004)
        """
        x1 = np.sqrt(8*π*self.Tie_c)
        a_ep = 1.09

        γep = (x1 + x1**2)/( 1 + a_ep*x1 + x1**2)
        # SETTING γ fit to 1 !!
        # return self.βv_Improved_Kelbg(self.Γei, r, self.Λei, γ=1 ) 
        return self.βv_Deutsch(self.Γei, r, self.Λei)

    def βvei_atomic(self, r, core_height=0):
        r_c  = self.r_c #3/5 r_s in linear n_b(r) model
        Λ, Γ = self.Λei, self.Γei
        one_over_r_cutoff = np.heaviside(r - r_c, 0.5)/r + core_height/r_c * np.heaviside(r_c - r,0.5)
        βvei_pseudopotential = Γ*one_over_r_cutoff
        return βvei_pseudopotential * ( 1 -  np.exp(-r/Λ) )

    def βvii(self, r):
        return self.Γii/r 

    def βv_Yukawa(self, r):
                return self.Γii/r * np.exp(-r*self.get_κ())
        