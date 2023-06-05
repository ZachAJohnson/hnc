import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

from math import isnan

from pandas import read_csv

from hnc import HNC_solver

from scipy.special import erfc

# Build Components for QSP's

m_e = 1
m_p = 1833.274
eV = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
Kelvin = 8.61732814974493e-5*eV #Similarly, 1 Kelvin = 3.167e-6... in natural units 
aB = 5.29177210903e-11 # Bohr radius in m

π = np.pi

class QSP_HNC():


    def __init__(self, Z, A, Zstar, Te, Ti, ri):
        self.Z = Z
        self.A = A
        self.Zstar = Zstar
        self.Te = Te
        self.Ti = Ti
        self.ri = ri

        self.initialize_physics(Z, A, Zstar, Te, Ti, ri)

    def initialize_physics(self, Z, A, Zstar, Te, Ti, ri):
        # [AU]

        self.ri = ri
        self.ni = self.n_from_rs(ri)
        
        self.βi   = 1/self.Ti

        self.m_i = m_p*self.A

        self.ne = self.Zstar*self.ni
        self.re = self.rs_from_n(self.ne)

        self.E_F = 1/(2*m_e) * (3*π**2 * self.ne)**(2/3)
        self.v_F = np.sqrt(2*self.E_F/m_e)
        # self.θ   = self.Te_c/self.E_F

        # self.lambda_TF = np.sqrt( self.Te_c / (4*π*self.ne)  )

        self.βe  = 1/self.Te
        # self.Λe  = np.sqrt(  self.βe*2*π /m_e )
        self.Λee  = 1/np.sqrt(π*m_e )/self.ri
        self.Λei  = 1/np.sqrt(2*π*m_e )/self.ri

        self.Tie = self.Tei_thermal()
        self.βie = 1/self.Tie
        
        # Λeff = np.sqrt(6*π) / np.sqrt( 3*Te/m_e + 0.6*v_F**2)/m_e

        print("Λee = {0:.3f}".format(self.Λee))
        print("Λei = {0:.3f}".format(self.Λei))
        self.Γee =  self.βe/self.ri 
        self.Γei = -self.Zstar*self.βie/self.ri
        self.Γii =  self.Zstar**2*self.βi/self.ri 

        self.Γ_matrix = np.array(  [[self.Γii,  self.Γei],
                                    [self.Γei,  self.Γee]])
 

        print("Γii={0:.1f}, Γie={1:.1f}, Γee={2:.1f} ".format(self.Γii, self.Γei, self.Γee))
        print("r_i={0:.1f}".format(self.ri))
        print("r_e={0:.1f}".format(self.re))
        # print("θ  ={0:.2e}".format(self.θ))

    @staticmethod
    def n_from_rs( rs):
        return 1/(4/3*π*rs**3)

    @staticmethod
    def rs_from_n(n):
        return (4/3*π*n)**(-1/3)

    def Tei_geometric(self):
        return np.sqrt(self.Te*self.Ti)

    def Tei_thermal(self):
        μ_ei = self.m_i*m_e/(m_e+self.m_i) #reduced mass
        Tie = μ_ei * (self.Te/m_e + self.Ti/self.m_i) #thermal velocity defined Tei
        return Tie

    def βv_Deutsch(self, Γ, r, Λe):
        # return Γ/r* ( 1 -  np.exp(-np.sqrt(2)π*r/Λe) ) # for Sarkas Λ
        return Γ/r* ( 1 -  np.exp(-r/Λe) )

    # Kelbg looks really weird??
    def βv_Kelbg(self, Γ,r, Λe):
        return Γ/r*( 1 - np.exp(-r**2/(Λe**2*2*π)) + r/Λe*erfc(r/Λe/np.sqrt(2*π)) )
        # return Γ/r*( 1 - np.exp(-2*π*r**2/Λe**2)+ np.sqrt(2)*π*r/Λe*erfc(np.sqrt(2*π)*r/Λe))

    def βv_Pauli(self, r, Λe):
        #Sarkas
        # return  np.log(2) * np.exp(-4*π* r**2 /( Λe**2))
        return -np.log(1 - 0.5*np.exp(-r**2/(2*π*Λe**2)) )
        

    # def βv_Pauli(r):
    #     # Jones + Murillo Eqn.
    #     return  -np.log(1-0.5*np.exp(-2*π*r**2/Λe_star**2))


    ######### Build Actual QSP's

    def βvee(self, r):
        return self.βv_Deutsch(self.Γee,r, self.Λee) + self.βv_Pauli(r,self.Λee)

    def βvei(self, r):
        return self.βv_Deutsch(self.Γei,r,self.Λei) #+ (Z-Zstar)*βv_Pauli(r)

    def βvei_atomic(self, r):
        r_c = 3/5 #3/5 r_s in linear n_b(r) model
        return np.heaviside(r - r_c,0.5) * self.βvei(r) + self.βvei(r_c)*np.heaviside(r_c - r,0.5)

    def βvii(self, r):
        return self.Γii/r 

    def βv_Yukawa(self, r):
                return self.Γii/r * np.exp(-r*self.ri/self.lambda_TF)
        