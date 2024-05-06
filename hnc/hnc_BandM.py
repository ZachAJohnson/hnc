import numpy as np
from scipy import fftpack
from scipy.optimize import newton, root

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from math import isnan

from pandas import read_csv

from scipy.optimize import minimize
from scipy.linalg import solve_sylvester, solve_continuous_lyapunov

from .constants import *



class Integral_Equation_Solver():
    def __init__(self, N_species, number_densities_in_rs, temperature_in_AU_list, u_matrix = None, dst_type=3, h_max=1e3,
       Γ_matrix = None, κ_screen = None, kappa_multiscale = 1.0,  R_max=25.0, N_bins=512, names=None, closure='hnc', use_U00_closure = False, use_U00_svt_correction = False):

        matrify = lambda numbers: np.array(numbers).reshape(N_species,N_species) 
        vectorfy = lambda numbers: np.array(numbers).reshape(N_species)

        self.N_species = N_species 
        self.rho = vectorfy(number_densities_in_rs)
        self.kappa_multiscale = kappa_multiscale
        self.R_max = R_max
        self.N_bins = N_bins
        self.dst_type = dst_type
        self.h_max=h_max
        self.T_list = np.array(temperature_in_AU_list)
        self.β_list = 1/self.T_list
        self.use_U00_closure = use_U00_closure # Whether or not to insert an explicit U_00. Necessary for non-equilibrium case. 
        self.use_U00_svt_correction = use_U00_svt_correction # We can correct the e-e term in BMOZ explicitly to get SVT.

        if np.all(self.T_list != self.T_list[0]):
            print("WARNING: Non-equilibrium system detected. You MUST put lowest mass particle at zero index. Currently all heavy particles need the same temperature.")

            if self.U00_closure is False:
                print("ERROR: U00_closure not set for non-equilibrium system. Nonsensical answer results.")


        self.I = np.eye(N_species)
        self.closure = closure # py or hnc
        
        self.make_k_r_spaces()
        self.initialize()

        if names is None:
            self.names = ['']*self.N_species
        else:
            self.names=names

        self.make_name_matrix()
        self.make_Γκ(Γ_matrix, κ_screen)
        self.initialize_u_matrix(u_matrix)

    def make_Γκ(self, Γ_matrix, κ_screen):
        if Γ_matrix is None:
            self.Γ_matrix = np.ones((self.N_species,self.N_species))
        else:
            self.Γ_matrix = np.array(Γ_matrix)
        if κ_screen is None:
            self.κ_screen = 1
        else:
            self.κ_screen = κ_screen

    @staticmethod
    def Bridge_function_OCP(x, Gamma ):
        """
        Computes OCP bridge function from 

        "Bridge functions and improvement on the hypernetted-chain approximation for classical one-component plasmas"
        by Hiroshi Iyetomi, Shuji Ogata, and Setsuo Ichimaru
        PHYSICAL REVIEW A VOLUME 46, NUMBER 2 15 JULY 1992

        Args:
            x: r/r_s radius over wigner-seitz ion sphere radius
            Gamma: Γ strong coupling parameter
        Returns:
            B_OCP(r)
        """
        if Gamma>5.5:
            b0 = 0.258 - 0.0612 *np.log(Gamma) + 0.0123*np.log(Gamma)**2 - 1/Gamma
            b1 = 0.0269 + 0.0318* np.log(Gamma) + 0.00814*np.log(Gamma)**2

            c1 =  0.498 - 0.280 *np.log(Gamma) + 0.0294 *np.log(Gamma)**2
            c2 = -0.412 + 0.219 *np.log(Gamma) - 0.0251 *np.log(Gamma)**2
            c3 = 0.0988 - 0.0534*np.log(Gamma) + 0.00682*np.log(Gamma)**2

            B_r = Gamma* (  -b0 + c1*x**4  + c2*x**6 + c3*x**8 )*np.exp(-b1/b0 * x**2) 
        else:
            B_r = 0
        return B_r

    @classmethod
    def Bridge_function_Yukawa(cls, x, Gamma, kappa ):
        """
        Computes Yukawa bridge function from 

        "Empirical bridge function for strongly coupled Yukawa systems"
        by William Daughton, Michael S. Murillo, and Lester Thode
        PHYSICAL REVIEW E VOLUME 61, NUMBER 2 FEBRUARY 2000

        Args:
            x: r/r_s radius over wigner-seitz ion sphere radius
            Gamma: Γ strong coupling parameter
            kappa: screening parameter (unitless since a/λs) 
        Returns:
            B_Yuk(r)
        """
        
        B_OCP_r = cls.Bridge_function_OCP(x, Gamma)
        B_Yuk_r = B_OCP_r*np.exp(-kappa**2/4)
        return B_Yuk_r

    def make_name_matrix(self):
        self.name_matrix = [[f'{a}-{b}' if a != b else a for b in self.names] for a in self.names]

    # Initilization methods
    def initialize(self):
        self.h_r_matrix  = np.zeros((self.N_species, self.N_species, self.N_bins))
        self.h_k_matrix  = np.zeros_like(self.h_r_matrix)
        self.γ_s_r_matrix = np.zeros_like(self.h_r_matrix)
        self.γ_s_k_matrix = np.zeros_like(self.h_r_matrix)
        self.u_r_matrix = np.zeros_like(self.h_r_matrix)
        self.U_k_matrix  = np.zeros_like(self.h_r_matrix)
        self.U_s_k_matrix= np.zeros_like(self.h_r_matrix)
        self.U_s_r_matrix= np.zeros_like(self.h_r_matrix)

        # self.initialize_u_matrix()
        # self.initialize_U_k()
        # self.set_U_matrix()

    def initialize_u_matrix(self, u_matrix):
        # Initialize to a Yukawa potential as a default
        if u_matrix is None:        
            u_matrix = self.Γ_matrix[:,:,np.newaxis]/self.r_array[np.newaxis, np.newaxis,:]*np.exp(- self.κ_screen * self.r_array[np.newaxis, np.newaxis,:])         # Regardless, need to set a number of arrays with this potential
        self.set_u_matrix(u_matrix)
            

    def initialize_U_k(self):
        """
        The polarization potential. This can be defined as U_ij = -δ^2 F^{ex}/δn_i δn_j for multi-temperature systems.
        """
        self.U_k_matrix[:,:,:] = self.u_k_matrix[:,:,:]
        self.U_r_matrix = self.FT_k_2_r_matrix(self.U_k_matrix)
        self.U_s_k_matrix = self.U_k_matrix - self.u_l_k_matrix
        self.U_s_r_matrix = self.FT_k_2_r_matrix(self.U_s_k_matrix)

    # R and K space grid and Fourier Transforms
    def make_k_r_spaces(self):
        
        if  self.dst_type==1: 
            self.r_array = np.linspace(0, self.R_max, num=self.N_bins+1)[1:]
            self.del_r = self.r_array[1]-self.r_array[0]
            self.k_array = np.array([π*(l+1)/self.R_max for l in range(self.N_bins)] ) #Type 1
            self.del_k = self.k_array[1]-self.k_array[0] 
        elif self.dst_type==2:
            self.r_array = np.linspace(0, self.R_max, num=self.N_bins+1)[1:]
            self.del_r = self.r_array[1]-self.r_array[0]
            self.r_array -= self.del_r/2 #So now theoretically r_array[1/2] would be zero
            self.k_array = np.array([π*(l+1)/self.R_max for l in range(self.N_bins)] ) #Type 1
            self.del_k = self.k_array[1]-self.k_array[0]
        elif self.dst_type==3:   
            self.r_array = np.linspace(0, self.R_max, num=self.N_bins+1)[1:]
            self.del_r = self.r_array[1]-self.r_array[0]
            self.k_array = np.array([π*(l+0.5)/self.R_max for l in range(self.N_bins)] ) #Type 3
            self.del_k = self.k_array[1]-self.k_array[0]
        elif self.dst_type==4:       
            self.r_array = np.linspace(0, self.R_max, num=self.N_bins+1)[1:]
            self.del_r = self.r_array[1]-self.r_array[0]
            self.r_array -= self.del_r/2 #So now theoretically r_array[1/2] would be zero
            self.k_array = np.array([π*(l+0.5)/self.R_max for l in range(self.N_bins)] ) #Type 3
            self.del_k = self.k_array[1]-self.k_array[0]        

        self.fact_r_2_k = 2 * np.pi * self.del_r
        self.fact_k_2_r = self.del_k / (4. * np.pi**2)
        self.dimless_dens = 3. / (4 * np.pi)

    def FT_r_2_k(self, input_array):
        from_dst = self.fact_r_2_k * fftpack.dst(self.r_array * input_array, type=self.dst_type)
        return from_dst / self.k_array

    
    def FT_k_2_r(self, input_array):
        from_idst = self.fact_k_2_r * fftpack.idst(self.k_array * input_array, type=self.dst_type)
        return from_idst / self.r_array

    
    def FT_r_2_k_matrix(self, f_r):
        N = f_r.shape[0]
        f_k = np.zeros((N, N, self.N_bins))
        f_k[:,:] = self.FT_r_2_k(f_r[:,:])
        return f_k
    
    def FT_k_2_r_matrix(self, f_k):
        N = f_k.shape[0]
        f_r = np.zeros((N, N, self.N_bins))
        f_r = self.FT_k_2_r(f_k)
        return f_r

    # Setters  
    def set_u_matrix(self, u_matrix):
        self.u_r_matrix = u_matrix
        self.split_u_matrix()
        self.set_u_k_matrices()
        self.initialize_U_k()


    def split_u_matrix(self):
        """
        Divides βu into long and short using exponent
        u_s_r = u_r * exp(- kappa_multiscale * r)
        u_l_r = u_r * ( 1 - exp(- kappa_multiscale * r) )
        s.t. u_s_r + u_l_r = u_r
        """
        self.u_s_r_matrix = self.u_r_matrix * np.exp(- self.kappa_multiscale * self.r_array)[np.newaxis, np.newaxis,:]
        self.u_l_r_matrix = self.u_r_matrix - self.u_s_r_matrix  # Equivalent definition for numerics    
    
    def set_u_k_matrices(self):
        self.u_s_k_matrix = self.FT_r_2_k_matrix(self.u_s_r_matrix)
        self.u_l_k_matrix = self.FT_r_2_k_matrix(self.u_l_r_matrix)
        self.u_k_matrix = self.FT_r_2_k_matrix(self.u_r_matrix)

    # def set_U_matrix(self):
    #     """
    #     Defines matrix of rho_i U_ij using initial assumption of diagonal
    #     """
    #     self.U_matrix = self.rho[np.newaxis,:,np.newaxis] * self.U_k_matrix
   

    # Matrix Manipulation
    def invert_matrix(self, A_matrix):
        """
        Calls routine to invert an NxN x N_bins matrix
        """
        A_inverse = np.zeros_like(A_matrix)#np.linalg.inv(A_matrix)

        if self.N_species==1:
            A_inverse = 1/A_matrix 
        elif self.N_species==2:
            det = A_matrix[0,0]*A_matrix[1,1] - A_matrix[1,0]*A_matrix[0,1]
            A_inverse[0,0] =  A_matrix[1,1]/det
            A_inverse[0,1] = -A_matrix[0,1]/det
            A_inverse[1,0] = -A_matrix[1,0]/det
            A_inverse[1,1] =  A_matrix[0,0]/det
        else:
            for k_index in range(self.N_bins):
                A_inverse[:,:, k_index] = np.linalg.inv(A_matrix[:,:,k_index])

        return A_inverse    

    def A_times_B(self,A,B):    
        """
        Multiplies N x N x N_bin
        """
        if self.N_species==1:
            product = A*B
        elif self.N_species==2:
            product = np.zeros_like(self.h_r_matrix)
            product[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            product[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            product[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            product[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] 
        else:
            product = np.einsum('ikm,kjm->ijm', A, B)
        return product
        
    def updater(self,old, new, alpha0 = 0.1):
        max_change = np.max(new/old)
        alpha  = np.min([alpha0,  alpha0/max_change])
        return old*(1-alpha) + new*alpha

    def get_S_k_matrix(self, h_k_matrix):
        tot_rho = np.sum(self.rho)
        x_array  = self.rho/tot_rho
        x_matrix = np.diag(self.rho)/tot_rho
        S_k_matrix = x_matrix[:,:,np.newaxis] + self.rho[:,np.newaxis,np.newaxis]*self.rho[np.newaxis,:,np.newaxis]/tot_rho*h_k_matrix
        return S_k_matrix

    def get_all_matrices_from_Usk(self, U_s_k_matrix):
        U_k_matrix   = U_s_k_matrix + self.u_l_k_matrix 
        U_s_r_matrix = self.FT_k_2_r_matrix(U_s_k_matrix)
        U_r_matrix   = U_s_r_matrix + self.u_l_r_matrix
        γ_s_k_matrix = self.get_γ_s_k_matrix(U_s_k_matrix)                           # 1. U_k, u_l_k -> γ_k   (Definition)
        γ_s_r_matrix = self.FT_k_2_r_matrix(γ_s_k_matrix) # γ_k        -> γ_r   (FT)     
        
        # potential of mean force    
        T_matrix = np.ones((self.N_species,self.N_species)) / self.β_list[1]
        T_matrix[0,0] = 1/self.β_list[0]
        βω_r_matrix = self.u_s_r_matrix - T_matrix[:,:,np.newaxis]*γ_s_r_matrix   
        
        heff_r_matrix  = self.get_heff_from_γs_Us(γ_s_r_matrix, U_s_r_matrix) # Get γs from OZ solution. γs_ij = βion h_ij + β_i Us_ij β_j 
        heff_k_matrix = self.FT_k_2_r_matrix(heff_r_matrix)
        h_r_matrix = self.get_h_matrix_from_heff_matrix(heff_r_matrix, k_space=False)
        h_r_matrix = np.where(h_r_matrix>self.h_max, self.h_max, h_r_matrix)
        h_r_matrix = np.where(h_r_matrix<-1, -1, h_r_matrix)
        h_k_matrix = self.FT_r_2_k_matrix(h_r_matrix)
        S_k_matrix = self.get_S_k_matrix(h_k_matrix)

        return U_k_matrix ,U_s_r_matrix ,U_r_matrix ,γ_s_k_matrix ,γ_s_r_matrix ,βω_r_matrix ,h_r_matrix ,h_r_matrix ,h_k_matrix, S_k_matrix, heff_r_matrix, heff_k_matrix
    
    def set_all_matrices_from_Usk(self, U_s_k_matrix):
        U_k_matrix ,U_s_r_matrix ,U_r_matrix ,γ_s_k_matrix ,γ_s_r_matrix ,βω_r_matrix ,h_r_matrix ,h_r_matrix ,h_k_matrix, S_k_matrix, heff_r_matrix, heff_k_matrix = self.get_all_matrices_from_Usk(U_s_k_matrix)
        self.U_k_matrix   = U_k_matrix
        self.U_s_r_matrix = U_s_r_matrix
        self.U_r_matrix   = U_r_matrix
        self.γ_s_k_matrix  = γ_s_k_matrix
        self.γ_s_r_matrix  = γ_s_r_matrix
        self.βω_r_matrix  = βω_r_matrix
        self.h_r_matrix   = h_r_matrix
        self.h_k_matrix   = h_k_matrix
        self.S_k_matrix   = S_k_matrix
        self.heff_r_matrix = heff_r_matrix
        self.heff_k_matrix = heff_k_matrix
        
    def get_heff_from_γs_Us(self, γ_s_matrix, U_s_matrix ):
        return (γ_s_matrix - self.β_list[:,np.newaxis,np.newaxis] * self.β_list[np.newaxis,:,np.newaxis]*U_s_matrix)/self.β_list[1]

    def Picard_U_s_k(self, old_U_s_k_matrix, new_U_s_k_matrix, alpha=0.1 ):
        return old_U_s_k_matrix*(1-alpha) + new_U_s_k_matrix*alpha


    def check_Usk00_closure(self, U_s_k_matrix):
        if self.use_U00_closure == True:
            U_s_k_matrix[0,0] = self.u_s_k_matrix[0,0]
            return U_s_k_matrix
        else:
            return U_s_k_matrix


    def total_err(self, U_s_k_matrix, verbose=False):
        U_s_k_matrix = self.check_Usk00_closure(U_s_k_matrix) # Does nothing if not needed. Otherwise, enforces low mass closure 
        γ_s_k_matrix = self.get_γ_s_k_matrix(U_s_k_matrix)  # 1. U_k, u_l_k -> γ_k   (Definition)
        h_k_HNC_matrix  = self.guess_h_k_matrix(γ_s_k_matrix)
        
        heff_k_OZ_matrix  = self.get_heff_from_γs_Us(γ_s_k_matrix, U_s_k_matrix ) # Get γs from OZ solution. γs_ij = βion h_ij + β_i Us_ij β_j 
        h_k_OZ_matrix = self.get_h_matrix_from_heff_matrix(heff_k_OZ_matrix)

        tot_eqn =  h_k_OZ_matrix - h_k_HNC_matrix
        if verbose == True:
            print("Species err: ", np.linalg.norm(tot_eqn, axis=(2))/np.sqrt(self.N_bins))
        if self.use_U00_closure==True:
            tot_eqn[0,0] = 0
        tot_err = np.linalg.norm(tot_eqn) /np.sqrt(self.N_bins*self.N_species**2)

        return tot_err

  
    def set_γ_s_k_matrix(self):
        """
        invert N_species x N_species  matrix equation to get γ_k = η_k - U_k o 
        For script h, or  η_ij = T_ion h_ij if i or j = ion. Else, unclear.
        """
        self.γ_s_k_matrix = self.get_γ_s_k_matrix(self.U_s_k_matrix)
        
    def get_γ_s_k_matrix(self, U_s_k_matrix):
        """
            Note this does NOT do the low-mass - low-mass (e-e) term correctly! This term does not have a closure as of yet.

        C_ij  = β_i U_ij β_j
        Cs_ij = C_ij - Cl_ij
        Cl_ij  = β_i u^L_ij β_j
        D_ij  = n_i U_ij β_j
        H_ij  = heff β_ion # For heff the script h in the paper/notes

        In terms of which we have the Boercker & More OZ equation.
        H_ij = -C_ij - H_ik D_kj

        Let γ = H + C, γs_k = H + Cs  
        H   = -C (1 + D)^-1
        γs  = (-Cl + Cs_k D ) (1 + D)^-1
        """ 
        U_s_k_matrix = self.check_Usk00_closure(U_s_k_matrix) # Does nothing if not needed. Otherwise, enforces low mass closure 

        U_k_matrix = U_s_k_matrix + self.u_l_k_matrix
        Cs_matrix  = self.β_list[:,np.newaxis,np.newaxis] * U_s_k_matrix * self.β_list[np.newaxis,:,np.newaxis] #    βi U_kj βj
        Cl_matrix  = self.β_list[:,np.newaxis,np.newaxis] * self.u_l_k_matrix * self.β_list[np.newaxis,:,np.newaxis]  # βi u_ij βj
        D_matrix   = self.rho[:,np.newaxis,np.newaxis]    * U_k_matrix * self.β_list[np.newaxis,:,np.newaxis] # n_k U_kj βj

        denominator  = self.invert_matrix(self.I[:,:,np.newaxis] + D_matrix)
        numerator    = -Cl_matrix + self.A_times_B( Cs_matrix, D_matrix ) 
        γ_s_k_matrix = self.A_times_B( numerator, denominator )
        
        return γ_s_k_matrix

 
    def get_heff_matrix_from_h_matrix(self, h_matrix, k_space=True):
        """
        For script h, or  heff_ij = h_ij if i or j = ion. 
        Approximate by ignoring <ne ne>/nene-1 term means heff_00 = βe/βi h_ee 
        Fully, heff_ee = βe/βi h_ee  + (1-βe/βi) f_ee for 
        Or,    h_ee = βi/βe heff_ee  + (1-βi/βe) f_ee for 
            f_ee = (<ne ne>/nene-1)
        Note, f_ee = n_i die^2 βe/(1 - ne βe dee) recovers SVT exactly!
        Instead, f_ee = n_i die^2 βe recovers the electronic part (i-i still wrong) of MASS OZ.
        """
        heff_matrix = h_matrix.copy()
        βe, βi = self.β_list[0:2] 
        ne, ni = self.rho[0:2]
        # Uei = self.U_s_k_matrix[0,1] + self.u_l_k_matrix[0,1]
        # Uee = self.U_s_k_matrix[0,0] + self.u_l_k_matrix[0,0]
        # Uii = self.U_s_k_matrix[1,1] + self.u_l_k_matrix[1,1]
        # D = (1+ne*βe*Uee)*(1+ni*βi*Uii) - ne*ni*βe*βi*Uei**2
        
        
        if k_space == True:
            hei = h_matrix[0,1]
            hii = h_matrix[1,1]
            if self.use_U00_svt_correction:
                fee = ni*hei**2/(1 + ni*hii)
            else:
                fee = np.zeros((self.N_bins))
            self.fee_k = fee
            self.fee_r = self.FT_k_2_r(fee)
        else:
            hei = self.FT_r_2_k(h_matrix[0,1])
            hii = self.FT_r_2_k(h_matrix[1,1])
            if self.use_U00_svt_correction:
                fee = ni*hei**2/(1 + ni*hii)
            else:
                fee = np.zeros((self.N_bins))
            self.fee_k = fee
            self.fee_r = self.FT_k_2_r(fee)
            fee = self.fee_r.copy()

        heff_matrix[0,0] = βe/βi * h_matrix[0,0] + (1-βe/βi)*fee
        return heff_matrix

    def get_h_matrix_from_heff_matrix(self, heff_matrix, k_space=True):
        """
        For script h, or  heff_ij = h_ij if i or j = ion. 
        Approximate by ignoring <ne ne>/nene-1 term means heff_00 = βe/βi h_ee 
        Fully, heff_ee = βe/βi h_ee  + (1-βe/βi) f_ee for 
        Or,    h_ee = βi/βe heff_ee  + (1-βi/βe) f_ee for 
            f_ee = (<ne ne>/nene-1)
        Note, f_ee = n_i die^2 βe/(1 - ne βe dee) recovers SVT exactly!
        Instead, f_ee = n_i die^2 βe recovers the electronic part (i-i still wrong) of MASS OZ.
        """
        h_matrix = heff_matrix.copy()
        βe, βi = self.β_list[0:2] 
        ne, ni = self.rho[0:2]
        # Uei = self.U_s_k_matrix[0,1] + self.u_l_k_matrix[0,1]
        # Uee = self.U_s_k_matrix[0,0] + self.u_l_k_matrix[0,0]
        # Uii = self.U_s_k_matrix[1,1] + self.u_l_k_matrix[1,1]

        # D = (1+ne*βe*Uee)*(1+ni*βi*Uii) - ne*ni*βe*βi*Uei**2
                
        if k_space == True:
            hei = h_matrix[0,1]
            hii = h_matrix[1,1]
            if self.use_U00_svt_correction:
                fee = ni*hei**2/(1 + ni*hii)
            else:
                fee = np.zeros((self.N_bins))
            self.fee_k = fee
            self.fee_r = self.FT_k_2_r(fee)
        else:
            hei = self.FT_r_2_k(h_matrix[0,1])
            hii = self.FT_r_2_k(h_matrix[1,1])
            if self.use_U00_svt_correction:
                fee = ni*hei**2/(1 + ni*hii)
            else:
                fee = np.zeros((self.N_bins))
            self.fee_k = fee
            self.fee_r = self.FT_k_2_r(fee)
            fee = self.fee_r.copy()
        
        h_matrix[0,0] = βi/βe * heff_matrix[0,0] + (1-βi/βe)*fee
        return h_matrix

    def guess_h_k_matrix(self, γ_s_k_matrix):
        if self.closure in ['HNC','hnc']:
            return self.guess_h_k_matrix_hnc(γ_s_k_matrix)
        elif self.closure in ['PY','py']:
            return self.guess_h_k_matrix_py(γ_s_k_matrix)

    def guess_h_k_matrix_hnc(self, γ_s_k_matrix): # HNC closure for Te!=Ti is only accurate for lower mass on right index. Meaning, of [[00,01],[10,11]] only lower triangle is accurate.
        # Get γs from OZ solution. γs_ij = βion heff_ij + β_i Us_ij β_j
    
        heff_k_matrix = self.get_heff_from_γs_Us(γ_s_k_matrix, self.U_s_k_matrix )
        h_k_matrix = self.get_h_matrix_from_heff_matrix(heff_k_matrix, k_space=True)
        h_r_matrix = self.FT_k_2_r_matrix(h_k_matrix)

        β_matrix = np.ones((self.N_species,self.N_species)) * self.β_list[1] # Te everywhere but 11
        β_matrix[0,:] = self.β_list[0]
        β_matrix[:,0] = self.β_list[0]

        h_r_matrix = -1 + np.exp(h_r_matrix + (self.U_s_r_matrix - self.u_s_r_matrix)*β_matrix[:,:,np.newaxis] ) # 2. γ_r,u_s_r  -> h_r   (HNC)   
        h_r_matrix = np.where(h_r_matrix>self.h_max, self.h_max, h_r_matrix)
        h_k_matrix = self.FT_r_2_k_matrix(h_r_matrix)
        return h_k_matrix


    def guess_U_s_k_matrix(self, U_s_k_matrix):
        γ_s_k_matrix = self.get_γ_s_k_matrix(U_s_k_matrix)  # 1. U_k, u_l_k -> γ_k   (Definition)
        h_k_matrix  = self.guess_h_k_matrix(γ_s_k_matrix)
        heff_k_matrix = self.get_heff_matrix_from_h_matrix(h_k_matrix)
        new_U_s_k_matrix = (γ_s_k_matrix - self.β_list[1]*heff_k_matrix) / self.β_list[np.newaxis,:,np.newaxis]/self.β_list[:,np.newaxis,np.newaxis] # γs_ij = βion heff_ij + β_i Us_ij β_j
        U_s_k_matrix = self.check_Usk00_closure(U_s_k_matrix) # Does nothing if not needed. Otherwise, enforces low mass closure 

        return new_U_s_k_matrix


    # Solver
    def HNC_solve(self, num_iterations=1e3, tol=1e-6, iters_to_wait=1e2, iters_to_use=3, alpha_Ng = 1e-3 ,
        alpha_Picard = 0.1, alpha_oz = 0., iters_to_check=10 ,verbose=False, min_iters=20, dont_check=False):
        """ 
        Integral equation solutions for the classical electron gas 
        J. F. Springer; M. A. Pokrant; F. A. Stevens, Jr.
        Crossmark: Check for Updates
        J. Chem. Phys. 58, 4863–4867 (1973)
        https://doi.org/10.1063/1.1679070

        Their N is my γ = h - c
        1. U_k, u_l_k -> γ_k   (Definition)
        2. γ_r,u_s_r  -> h_r   (HNC)
        3. h_r, γ_r   -> U_s_r (Ornstein-Zernicke)
        4. U_s, u_l   -> U_r_k (Definition)
        Args: 
            species_left: tuple specifying which species to solve for
        Returns:
            None
        """

        converged = 1
        decreasing = True
        iteration = 1
        self.h_r_matrix_list, self.U_s_k_matrix_list = [], []
        self.h_r_matrix_list.append(self.h_r_matrix.copy())            
        self.U_s_k_matrix_list.append(self.U_s_k_matrix.copy())
        initial_error = self.total_err(self.U_s_k_matrix)
        print("0: Initial condition Total Error: {1:.3e}".format(iteration, initial_error))
        self.tot_err_list  = [initial_error]
        while converged!=0:
            old_U_s_k_matrix = self.U_s_k_matrix.copy()
            if iteration < iters_to_wait: #Picard at first
                self.U_s_k_matrix = self.check_Usk00_closure(self.U_s_k_matrix)
                guess_U_s_k_matrix = self.guess_U_s_k_matrix(self.U_s_k_matrix)
                self.U_s_k_matrix = self.Picard_U_s_k(self.U_s_k_matrix, guess_U_s_k_matrix, alpha=alpha_Picard )
                self.U_s_k_matrix = self.check_Usk00_closure(self.U_s_k_matrix)
            elif iteration == iters_to_wait:
                best_run_index = np.argmin(self.tot_err_list)
                self.U_s_k_matrix = self.U_s_k_matrix_list[best_run_index]
                print("Starting Ng loop, using best index so far: ", best_run_index)
                self.h_r_matrix_list = self.h_r_matrix_list[:best_run_index]
                self.U_s_k_matrix_list = self.U_s_k_matrix_list[:best_run_index]
                self.tot_err_list = self.tot_err_list[:best_run_index]
                iteration+=1
                continue
            else:
                self.U_s_k_matrix = self.Ng_U_s_k(num_to_use=iters_to_use, alpha = alpha_Ng)

            self.set_all_matrices_from_Usk(self.U_s_k_matrix)

            self.h_r_matrix_list.append(self.h_r_matrix.copy())            
            self.U_s_k_matrix_list.append(self.U_s_k_matrix.copy())
            
            err_c = np.linalg.norm(old_U_s_k_matrix - self.U_s_k_matrix) / np.sqrt(self.N_bins*self.N_species**2)
            
            actual_tot_err = self.total_err(self.U_s_k_matrix)
            self.tot_err_list.append(actual_tot_err)
            if iteration%1==0 and verbose==True:
                print("{0}: Change in U_r: {1:.3e}, Total Error: {2:.3e}".format(iteration, err_c, actual_tot_err))
            
            if dont_check == False:
                if len(self.tot_err_list) > iters_to_check:
                    Δ_err = np.array(self.tot_err_list[-iters_to_check:-2]) - np.array(self.tot_err_list[-iters_to_check + 1:-1])
                    if all( Δ_err < 0 ):
                        decreasing = False
                        print("QUIT: Last {0} iterations error has been increasing".format(iters_to_check))
                        converged=2
                        break

                if iteration >  200 and actual_tot_err > 1e2:
                    Δ_err = np.array(self.tot_err_list[-iters_to_check:-2]) - np.array(self.tot_err_list[-iters_to_check + 1:-1])
                    if all( Δ_err > 0 ):
                        "do nothing"
                    else:
                        print("QUIT: Large Error at many iterations, and error not decreasing.")
                        converged=3
                        break
                # if np.isinf(hnc_err)==True and iteration>50 :
                #     print("QUIT: HNC error infinite.")
                #     break
                
                if np.isinf(actual_tot_err)==True and iteration>min_iters :
                    print("QUIT: Total error infinite.")
                    break

                if isnan(err_c):
                    print("QUIT: U_r is nan.")
                    break

            if iteration > num_iterations:
                converged=1
                print("Warning: hnc_solver did not converge within the specified number of iterations.")                
                break

            if actual_tot_err < tol:
                converged = 0

            iteration += 1
        best_run_index = np.argmin(self.tot_err_list)
        self.U_s_k_matrix = self.U_s_k_matrix_list[best_run_index]# + self.u_l_k_matrix
        self.set_all_matrices_from_Usk(self.U_s_k_matrix)
        self.final_Picard_err = self.tot_err_list[best_run_index]
        
        self.h_r_matrix_list = self.h_r_matrix_list[:best_run_index+1]
        self.U_s_k_matrix_list = self.U_s_k_matrix_list[:best_run_index+1]
        self.tot_err_list = self.tot_err_list[:best_run_index+1]

        print("Exiting status {0}, reverting to best index so far: {1}".format(converged, best_run_index))
        print("Final iter:{0}, Total Error: {1:.3e}".format(iteration, self.final_Picard_err))

        return converged

    # HNC Inverter
    @staticmethod
    def remove_species(matrix, species):
        """
        Removes a specific species column and row from arrays
        """
        matrix = np.delete(matrix, species, axis=0)
        matrix = np.delete(matrix, species, axis=1)
        return matrix

    def initialize_effective(self):
        # self.Seff_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.ceff_k_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.ceff_r_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.βueff_r_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))

    def invert_hnc_OZ(self, removed_species):
        """ 
        Takes N species results and inverts it to find single effective v_ii for one species
        """
        self.Neff_species = self.N_species - len(removed_species)
        # First get new effective h matrix
        self.rhoeff        = np.delete(self.rho, removed_species)
        self.βωeff_r_matrix   = self.remove_species(self.βω_r_matrix, removed_species)
        self.heff_r_matrix = -1 + np.exp(-self.βωeff_r_matrix)  #self.remove_species(self.h_r_matrix, removed_species)
        self.heff_k_matrix = self.FT_r_2_k_matrix(self.heff_r_matrix)

        #Initialize other matrices to new size
        self.initialize_effective()

        # EXACT Ornstein-Zernike!
        I_plus_h_rho_inverse = self.invert_matrix(self.I[:self.Neff_species,:self.Neff_species,np.newaxis] + self.heff_k_matrix*self.rhoeff[:,np.newaxis,np.newaxis])
        # I_plus_h_rho_inverse = 1/(self.I[:self.Neff_species,:self.Neff_species, np.newaxis] + self.heff_k_matrix*self.rhoeff[:,np.newaxis,np.newaxis])
        self.ceff_k_matrix  =  self.A_times_B(I_plus_h_rho_inverse, self.heff_k_matrix)
        self.ceff_r_matrix = self.FT_k_2_r_matrix(self.ceff_k_matrix)

        # Approximate with HNC
        self.βueff_r_matrix   = self.heff_r_matrix - self.ceff_r_matrix + self.βωeff_r_matrix

    def get_symmetric_from_upper(self, upper_list):
        upper_indcs = np.triu_indices(self.N_species,k=1)
        upper_indcs_withdiag = np.triu_indices(self.N_species)
        lower_indcs = np.tril_indices(self.N_species, k=-1)

        symmetriU_matrix = np.zeros( (self.N_species, self.N_species) )
        symmetriU_matrix[upper_indcs_withdiag] = upper_list
        symmetriU_matrix[lower_indcs] = symmetriU_matrix[upper_indcs]
        return symmetriU_matrix

    def get_upper(self, symmetriU_matrix):
        upper_indcs_withdiag = np.triu_indices(self.N_species)        
        return symmetriU_matrix[upper_indcs_withdiag]

    def invert_SVT(self, h_k_matrix, U_s_k_matrix_guess, verbose=False):
        """
        Inverts SVT OZ to get U_s_k from h_k
        """
        
        U_s_k_matrix = np.zeros_like(self.h_k_matrix)
        for k_i in range(self.N_bins):
            u_l_k_matrix_single_k = self.u_l_k_matrix[:,:,k_i]
            h_k_matrix_single_k = h_k_matrix[:,:,k_i]

            def oz_f_to_min(U_s_k_upper_single_k):
                U_s_k_matrix_single_k = self.get_symmetriU_from_upper(U_s_k_upper_single_k)
                oz_matrix = self.SVT_matrix_eqn_at_single_k( h_k_matrix_single_k, U_s_k_matrix_single_k, u_l_k_matrix_single_k )
                return self.get_upper(oz_matrix)
            
            upper_U_s_k_guess = self.get_upper(U_s_k_matrix_guess[:,:,k_i])
            sol = root(oz_f_to_min, upper_U_s_k_guess) 
            U_s_k_matrix[:,:,k_i] = self.get_symmetriU_from_upper(sol.x)
            if verbose:
                print("Invert SVT: ", sol.success, sol.message)
        return U_s_k_matrix
        

    ############# PLOTTING #############
    def plot_species(self, species_nums, fig=None, ax=None):
        if fig is None and axs is None:
            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16,10))

        self.U_r_matrix = self.FT_k_2_r_matrix(self.U_k_matrix)
        fig.suptitle("Species: " + self.name_matrix[species_nums[0]][species_nums[1]] ,fontsize=20)
        # Top Left
        axs[0,0].plot(self.r_array, self.U_r_matrix[species_nums],'--.')
        axs[0,0].set_xlabel("r",fontsize=20)
        axs[0,0].set_ylabel("c(r)",fontsize=20)
        axs[0,0].set_title("Direct Correlation function",fontsize=20)
        axs[0,0].set_yscale('symlog',linthresh=0.1)
        
        # Top Right
        axs[0,1].plot(self.k_array, self.U_k_matrix[species_nums],'--.')
        axs[0,1].set_xlabel("k",fontsize=20)
        axs[0,1].set_ylabel("c(k)",fontsize=20)
        axs[0,1].set_title("Fourier Direct Correlation function",fontsize=20)
        axs[0,1].set_yscale('symlog',linthresh=0.1)

        # Bottom Left
        axs[1,0].plot(self.r_array, self.h_r_matrix[species_nums]+1,'--.')
        axs[1,0].set_xlabel(r"$r/r_s$",fontsize=20)
        axs[1,0].set_ylabel(r"$g(r/r_s)$",fontsize=20)
        axs[1,0].set_title("Radial distribution function",fontsize=20)
        # axs[1,0].set_ylim(0,10)
        # axs[1,0].set_yscale('log')#,linthresh=0.1)
        
        #Bottom Right
        axs[1,1].plot(self.r_array, np.exp(self.γ_s_r_matrix + self.u_l_r_matrix)[species_nums] ,'--.')
        axs[1,1].set_xlabel("r/r_s",fontsize=20)
        axs[1,1].set_ylabel(r"$e^\gamma(r)$",fontsize=20)
        axs[1,1].set_title(r"$n=n_{ideal}(r) e^{\gamma(r)}$",fontsize=20)
        axs[1,1].set_yscale('symlog',linthresh=0.1)

        for ax in axs.flatten():
            ax.tick_params(labelsize=20)
            ax.set_xscale('log')

        plt.tight_layout()
        # #plt.show()
        return fig, axs

    def plot_convergence_uex(self, ax):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
        ax.plot(self.u_ex_list)
        ax.set_title( "Excess Potential Energy Density" ,fontsize=15)
        # ax.set_ylim(0, np.max(  [np.max(self.h_r_matrix_list[k][i,j]+1), 5]) )
        # ax.set_xlim(0,41)
        ax.tick_params(labelsize=20)
        ax.set_xlabel("Iteration",fontsize=20)
        
        ax.set_ylabel(r"$u_{\rm ex} r_s^3$ [A.U.]",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        #plt.show()
        return fig, axs

    def plot_species_convergence_g(self, n_slices=4, fig=None, axs=None):
        if fig is None and axs is None:
            fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))
        fig.suptitle("Radial Distribution Convergence Blue to Red" ,fontsize=20)
        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_r_matrix_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.r_array, self.h_r_matrix_list[k][i,j]+1, color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) ,fontsize=15)
                    axs[i,j].set_ylim(0, np.max(  [np.max(self.h_r_matrix_list[k][i,j]+1), 5]) )
                    axs[i,j].set_xlim(0,4)
                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$r/r_s$",fontsize=20)
                    if np.max(self.h_r_matrix_list[k][i,j]+1) > 5:
                        axs[i,j].set_yscale('symlog',linthresh=10)

            axs[i,0].set_ylabel(r"$g(r/r_s)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)

        #plt.show()
        return fig, axs

    def plot_species_convergence_c(self, n_slices=4, fig=None, axs=None):
        if fig is None and axs is None:
            fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))
        fig.suptitle("Direct Correlation Function Convergence Blue to Red" ,fontsize=20)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_r_matrix_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.r_array, self.FT_k_2_r_matrix( (self.U_s_k_matrix_list - self.u_l_k_matrix)[k][i,j]), color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) ,fontsize=15)
                    axs[i,j].set_xlim(0,4)

                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$r/r_s$",fontsize=20)
                    axs[i,j].set_yscale('symlog',linthresh=0.1)

            axs[i,0].set_ylabel(r"$c(r/r_s)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        #plt.show()
        return fig, axs

    def plot_species_convergence_ck(self, n_slices=4, fig=None, axs=None):
        if fig is None and axs is None:
            fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))
        fig.suptitle("Direct Correlation Function Convergence Blue to Red" ,fontsize=20)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_r_matrix_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.k_array, self.U_s_k_matrix_list[k][i,j] - self.u_l_k_matrix[i,j], color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) ,fontsize=15)
                    # axs[i,j].set_xlim(0,10)
                    axs[i,j].set_xscale('log')

                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$k $",fontsize=20)
                    axs[i,j].set_yscale('symlog',linthresh=0.1)

            axs[i,0].set_ylabel(r"$c(k)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        #plt.show()
        return fig, axs

    def plot_species_convergence_Usk(self, n_slices=4, fig=None, axs=None):
        if fig is None and axs is None:
            fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))

        fig.suptitle("Direct Correlation Function Convergence Blue to Red" ,fontsize=20)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_r_matrix_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.k_array, self.U_s_k_matrix_list[k][i,j] , color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) ,fontsize=15)
                    # axs[i,j].set_xlim(0,10)
                    axs[i,j].set_xscale('log')

                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$k $",fontsize=20)
                    axs[i,j].set_yscale('symlog',linthresh=0.1)

            axs[i,0].set_ylabel(r"$U_s(k)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        #plt.show()
        return fig, axs

    def plot_g_grid(self, fig=None, axs=None):
        if fig is None and axs is None:
            fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(8*self.N_species,4*self.N_species))
        fig.suptitle("Radial Distribution Function for all Species",fontsize=20,y=1)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        for i in range(self.N_species):
            for j in range(self.N_species):
                axs[i,j].plot(self.r_array, self.h_r_matrix[i,j]+1)
                axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) ,fontsize=15)
                axs[i,j].set_ylim(0,2)
                axs[i,j].tick_params(labelsize=20)
                axs[-1,j].set_xlabel(r"$r/r_s$",fontsize=20)

            axs[i,0].set_ylabel(r"$g(r/r_s)$",fontsize=20)
                

        for ax in axs.flatten():
            ax.tick_params(labelsize=15)
        
        plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95], pad=0.4, w_pad=5.0, h_pad=5.0)
        #plt.show()
        return fig, axs

    def plot_g_all_species(self, data_to_compare=None, data_names=None, gmax=None, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,8))
        fig.suptitle("Radial Distribution Function for all Species", fontsize=20, y=1)

        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.r_array, self.h_r_matrix[i,j]+1,'--.', label=self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) )

        if data_to_compare:
            for file_name, label in zip(data_to_compare, data_names):
                r_datas, g_datas = np.array(read_csv(file_name, delimiter=',', header=1)).T
                ax.plot(r_datas, g_datas, '-', label=label)
                ax.legend(fontsize=15)

        if gmax:
            ax.set_ylim(0, gmax)

        ax.tick_params(labelsize=20)
        ax.set_xlabel(r"$r/r_s$", fontsize=20)
        ax.set_xlim(self.del_r, 10)
        ax.set_ylabel(r"$g(r/r_s)$", fontsize=20)
        ax.set_ylim(0, 5)
        ax.tick_params(labelsize=15)
        ax.legend(fontsize=20, loc = 'upper left')

        # Creating the inset plot
        # axins = inset_axes(ax, width='30%', height='30%', loc='upper right')
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        for i in range(self.N_species):
            for j in range(self.N_species):
                axins.plot(self.r_array, self.h_r_matrix[i,j]+1)
        
        # axins.set_xscale('log')
        axins.set_yscale('log')
        axins.set_xlim(0, 1)
        axins.set_ylim(1e-2, np.max(self.h_r_matrix+1)*2)
        axins.tick_params(labelsize=10)
        axins.set_xlabel(r"$r/r_s$", fontsize=12)
        axins.set_ylabel(r"$g(r/r_s)$", fontsize=12)
        
        # plt.tight_layout()
        #plt.show()
        return fig, ax
    
    def plot_ck_all_species(self, data_to_compare=None, data_names=None, gmax=None, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,8))
        fig.suptitle("Radial Distribution Function for all Species", fontsize=20, y=1)

        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.k_array, self.U_k_matrix[i,j]+1,'--.', label=self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) )

        if data_to_compare:
            for file_name, label in zip(data_to_compare, data_names):
                r_datas, g_datas = np.array(read_csv(file_name, delimiter=',', header=1)).T
                ax.plot(r_datas, g_datas, '-', label=label)
                ax.legend(fontsize=15)

        if gmax:
            ax.set_ylim(0, gmax)

        ax.tick_params(labelsize=20)
        ax.set_xlim(self.del_r, 10)
        ax.set_xlabel(r"$k/k_s$", fontsize=20)
        ax.set_ylabel(r"$c(k/k_s)$", fontsize=20)
        ax.set_yscale('symlog', linthresh=0.1)
        ax.tick_params(labelsize=15)
        ax.legend(fontsize=20)
        
        plt.tight_layout()
        #plt.show()
        return fig, ax

    def plot_Usk_all_species(self, data_to_compare=None, data_names=None, gmax=None, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,8))
        fig.suptitle("Radial Distribution Function for all Species", fontsize=20, y=1)

        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.k_array, self.U_s_k_matrix[i,j]+1,'--.', label=self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) )

        if data_to_compare:
            for file_name, label in zip(data_to_compare, data_names):
                r_datas, g_datas = np.array(read_csv(file_name, delimiter=',', header=1)).T
                ax.plot(r_datas, g_datas, '-', label=label)
                ax.legend(fontsize=15)

        if gmax:
            ax.set_ylim(0, gmax)

        ax.tick_params(labelsize=20)
        ax.set_xlim(self.del_r, 10)
        ax.set_xlabel(r"$k/k_s$", fontsize=20)
        ax.set_ylabel(r"$U_s(k/k_s)$", fontsize=20)
        ax.set_yscale('symlog', linthresh=0.1)
        ax.tick_params(labelsize=15)
        ax.legend(fontsize=20)
        
        plt.tight_layout()
        #plt.show()
        return fig, ax

    def plot_g_vs_murillo(self, gmax=None):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8*self.N_species,4*self.N_species))
        fig.suptitle("Radial Distribution Function for all Species",fontsize=20,y=1)

        labels=[r"$g_{ii}$",r"$g_{ie}$",r"$g_{ei}$", r"$g_{ee}$"]
        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.r_array, self.h_r_matrix[i,j]+1,'--.', label=labels[i*2+j])
        
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = [colors[0], colors[2], colors[3]]
        file_names=["./MM_HNC/gii.out","./MM_HNC/gei.out","./MM_HNC/gee.out"]
        labels = [r"MM $g_{ii}$",r"MM $g_{ei}$", r"MM $g_{ee}$"]
        for file_name, label, color in zip(file_names, labels, colors):
            r_datas, g_datas = np.array(read_csv(file_name,delim_whitespace=True,header=None)).T
            ax.plot(r_datas, g_datas,'-', label=label, color=color)
            ax.legend(fontsize=15)


        if gmax != None:
            ax.set_ylim(0,gmax)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        ax.tick_params(labelsize=20)
        ax.set_xlabel(r"$r/r_s$",fontsize=20)
        ax.set_xlim(self.del_r,10)
        ax.set_ylabel(r"$g(r/r_s)$",fontsize=20)
        # ax.set_yscale('symlog',linthresh=0.1)
        ax.set_xscale('log')
        ax.tick_params(labelsize=15)
        
        plt.tight_layout()#rect=[0.03, 0.03, 0.95, 0.95], pad=0.4, w_pad=5.0, h_pad=5.0)
        plt.legend(fontsize=20)
        plt.grid()
        #plt.show()

    def plot_u_all_species(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,6))
            print("axs")
        fig.suptitle("Potential Energy between all Species",fontsize=20,y=1)

        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.r_array, self.u_r_matrix[i,j], label=self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Γ_matrix[i][j]) )
        
        # ax.set_ylim(0,4)
        ax.set_yscale('symlog',linthresh=1)
        ax.set_xscale('log')
        ax.tick_params(labelsize=20)
        ax.set_xlabel(r"$r/r_s$",fontsize=20)
        ax.set_xlim(0,10)
        ax.set_ylabel(r"$\beta u(r/r_s)$",fontsize=20)
        ax.tick_params(labelsize=15)
            
        plt.tight_layout()#rect=[0.03, 0.03, 0.95, 0.95], pad=0.4, w_pad=5.0, h_pad=5.0)
        plt.legend(fontsize=15)
        # #plt.show()
        return fig, ax

if __name__ == "__main__":
    N_species = 3
    Gamma = np.array(  [[1,  0.1,  3],
                        [0.1,  5,   3],
                        [3,    3,   5]])
    names = ["Ion-1","Ion-2", "Electron", ] 
    kappa = 3
    rho = np.array([  3/(4*np.pi), 3/(4*np.pi), 3/(4*np.pi)  ])
    hnc = hnc_solver(N_species, Gamma=Gamma, kappa=kappa, tol=3e-1, alpha=1e-3, rho = rho, num_iterations=int(1e4), R_max=10, N_bins=1000, names=names)

    self.hnc_solve()
    # print(self.U_r_matrix)

    h_r_matrix = self.h_r_matrix
    U_r_matrix = self.U_r_matrix

    # To get the radial distribution function g(r), you can add 1 to the h_r_matrix:
    g_r_matrix = h_r_matrix + 1

    # To visualize the results, you can use matplotlib to plot the g(r) function:
    # self.plot_species((0,0))
    # self.plot_species_convergence((0,0))
    self.plot_g_all_species()
    
   

