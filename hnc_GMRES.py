import numpy as np
from scipy import fftpack
from scipy.sparse.linalg import gmres

import matplotlib.pyplot as plt

from math import isnan

from pandas import read_csv

from scipy.optimize import minimize
from scipy.linalg import solve_sylvester, solve_continuous_lyapunov

π = np.pi


class HNC_solver():
    def __init__(self, N_species, Gamma, rho, temps, masses, dst_type=3, oz_method='standard',kappa = 1.0, kappa_multiscale = 1.0, tol=1e-4, num_iterations=1000, R_max=25.0, N_bins=512, names=None):
        self.N_species = N_species
        self.Gamma = Gamma
        self.rho = rho
        self.Temp_list = temps
        self.mass_list = masses
        self.kappa = kappa
        self.kappa_multiscale = kappa_multiscale
        self.num_iterations = num_iterations
        self.tol = tol
        self.R_max = R_max
        self.N_bins = N_bins
        self.dst_type = dst_type
        self.oz_method = oz_method

        self.Temp_matrix = (self.mass_list[:,np.newaxis]*self.Temp_list[np.newaxis,:] + self.mass_list[np.newaxis,:]*self.Temp_list[:,np.newaxis])/(self.mass_list[:,np.newaxis] + self.mass_list[np.newaxis,:])
        self.mass_matrix = (self.mass_list[:,np.newaxis]*self.mass_list[np.newaxis,:])/(self.mass_list[:,np.newaxis] + self.mass_list[np.newaxis,:])

        self.I = np.eye(N_species)
        
        self.make_k_r_spaces()
        self.initialize()

        if names!=None:
            self.names=names
            self.make_name_matrix()

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
        b0 = 0.258 - 0.0612 *np.log(Gamma) + 0.0123*np.log(Gamma)**2 - 1/Gamma
        b1 = 0.0269 + 0.0318* np.log(Gamma) + 0.00814*np.log(Gamma)**2

        c1 =  0.498 - 0.280 *np.log(Gamma) + 0.0294 *np.log(Gamma)**2
        c2 = -0.412 + 0.219 *np.log(Gamma) - 0.0251 *np.log(Gamma)**2
        c3 = 0.0988 - 0.0534*np.log(Gamma) + 0.00682*np.log(Gamma)**2

        B_r = Gamma* (  -b0 + c1*x**4  + c2*x**6 + c3*x**8 )*np.exp(-b1/b0 * x**2) 
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
        self.γs_r_matrix = np.zeros_like(self.h_r_matrix)
        self.γs_k_matrix = np.zeros_like(self.h_r_matrix)
        self.βu_r_matrix = np.zeros_like(self.h_r_matrix)
        self.c_k_matrix  = np.zeros_like(self.h_r_matrix)
        self.c_s_k_matrix= np.zeros_like(self.h_r_matrix)
        self.c_s_r_matrix= np.zeros_like(self.h_r_matrix)

        self.initialize_βu_matrix()
        self.initialize_c_k()
        self.set_C_matrix()

    def initialize_guess(self):
        self.h_r_matrix  = -1 + np.exp(-self.βu_r_matrix)
        self.h_k_matrix  = self.FT_r_2_k_matrix(self.h_r_matrix)
        self.ρh_k_matrix = self.rho[:,np.newaxis,np.newaxis]*self.h_k_matrix
            
        self.c_k_matrix  =  self.invert_matrix(self.I[:,:,np.newaxis] + self.ρh_k_matrix)*self.h_k_matrix

        self.c_s_k_matrix = self.c_k_matrix + self.βu_l_k_matrix
        self.c_s_r_matrix = self.FT_k_2_r_matrix(self.c_s_k_matrix)

    def initialize_βu_matrix(self):
        self.set_βu_matrix(self.Gamma[:,:,np.newaxis]*(np.exp(- self.kappa * self.r_array) / self.r_array)[np.newaxis, np.newaxis,:])

    def initialize_c_k(self):
        self.c_k_matrix[:,:,:] = -self.βu_k_matrix[:,:,:]
        self.c_r_matrix = self.FT_k_2_r_matrix(self.c_k_matrix)
        self.c_s_k_matrix = self.c_k_matrix + self.βu_l_k_matrix
        self.c_s_r_matrix = self.FT_k_2_r_matrix(self.c_s_k_matrix)
        # self.c_r_matrix   = self.c_s_k_matrix - self.βu_l_k_matrix

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
        from_dst = self.fact_r_2_k * fftpack.dst(self.r_array * input_array, type=3)
        return from_dst / self.k_array

    def FT_k_2_r(self, input_array):
        from_idst = self.fact_k_2_r * fftpack.idst(self.k_array * input_array, type=3)
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
    def set_βu_matrix(self, βu_matrix):
        self.βu_r_matrix = βu_matrix
        self.split_βu_matrix()
        self.set_βu_k_matrices()

    def split_βu_matrix(self):
        """
        Divides βu into long and short using exponent
        βu_s_r = βu_r * exp(- kappa_multiscale * r)
        βu_l_r = βu_r * ( 1 - exp(- kappa_multiscale * r) )
        s.t. βu_s_r + βu_l_r = βu_r
        """
        self.βu_s_r_matrix = self.βu_r_matrix * np.exp(- self.kappa_multiscale * self.r_array)[np.newaxis, np.newaxis,:]
        self.βu_l_r_matrix = self.βu_r_matrix - self.βu_s_r_matrix  # Equivalent definition for numerics    
    
    def set_βu_k_matrices(self):
        self.βu_s_k_matrix = self.FT_r_2_k_matrix(self.βu_s_r_matrix)
        self.βu_l_k_matrix = self.FT_r_2_k_matrix(self.βu_l_r_matrix)
        self.βu_k_matrix = self.FT_r_2_k_matrix(self.βu_r_matrix)

    def set_C_matrix(self):
        """
        Defines matrix of rho_i c_ij using initial assumption of diagonal
        """
        self.C_matrix = self.rho[np.newaxis,:,np.newaxis] * self.c_k_matrix
   
    def set_γs_k_matrix(self):
        """
        invert N_species x N_species  matrix equation to get γ_k = h_k - c_k
        Two methods implemented, normal OZ and SVT-OZ
        """
        self.γs_k_matrix = self.get_γs_k_matrix(self.c_s_k_matrix)
        

    def get_γs_k_matrix(self, c_s_k_matrix):
        if self.oz_method=='standard':
            γs_k_matrix = self.get_γs_k_matrix_standard(c_s_k_matrix)
        elif self.oz_method in ['svt','SVT']:
            γs_k_matrix = self.get_γs_k_matrix_SVT(c_s_k_matrix)
        return γs_k_matrix

    def get_γs_k_matrix_standard(self, c_s_k_matrix):
        """
        invert N_species x N_species  matrix equation to get γ_k = h_k - c_k
        γs_k = (I-C)^-1 (C c_k_s - u_s_k)
        """
        c_k_matrix = c_s_k_matrix - self.βu_l_k_matrix
        C_matrix = self.rho[np.newaxis,:,np.newaxis] * c_k_matrix
        denominator = self.invert_matrix(self.I[:,:,np.newaxis] - C_matrix)
        numerator   = self.A_times_B(C_matrix, c_s_k_matrix ) -  self.βu_l_k_matrix 
        γs_k_matrix = self.A_times_B(denominator, numerator )
        return γs_k_matrix
        # print("asymm γs: ", set((self.γs_k_matrix[0,1,:]/self.γs_k_matrix[1,0,:]).flatten()))
  

    def get_γs_k_matrix_SVT(self, c_s_k_matrix):
        """
        Uses SVT method of OZ
        γ = h-c 
        
        (V+D)γs + γs(V+E) = U - D cs - cs E 
        """
        c_k_matrix = c_s_k_matrix - self.βu_l_k_matrix
        V = np.diag(self.Temp_list/self.mass_list) 
        D = self.rho[np.newaxis,:,np.newaxis]*self.Temp_matrix[:,:,np.newaxis]/self.mass_list[:,np.newaxis, np.newaxis] * c_k_matrix
        E = self.rho[:,np.newaxis,np.newaxis]*self.Temp_matrix[:,:,np.newaxis]/self.mass_list[np.newaxis, :, np.newaxis] * c_k_matrix
        U = - self.βu_l_k_matrix * self.Temp_matrix[:,:,np.newaxis]/self.mass_matrix[:,:,np.newaxis]
        
        A = V[:,:,np.newaxis] - D
        B = V[:,:,np.newaxis] - E
        C = U + self.A_times_B(D, c_s_k_matrix) + self.A_times_B(c_s_k_matrix, E)
    
        γs_k_matrix = np.zeros_like(self.γs_k_matrix)
        for k in range(self.N_bins):
            γs_k_matrix[:,:,k]   = solve_sylvester(A[:,:,k], B[:,:,k], C[:,:,k])
            # γs_k_matrix[:,:,k] = solve_continuous_lyapunov(A[:,:,k], C[:,:,k])
        return γs_k_matrix
        
    # Matrix Manipulation
    def invert_matrix(self, A_matrix):
        """
        Calls routine to invert an NxN x N_bins matrix
        """

        A_inverse = np.zeros_like(A_matrix)#np.linalg.inv(A_matrix)

        for k_index in range(self.N_bins):
            A_inverse[:,:, k_index] = np.linalg.inv(A_matrix[:,:,k_index])

        return A_inverse    

    def A_times_B(self,A,B):    
        """
        Multiplies N x N x N_bin
        """
        product = np.einsum('ikm,kjm->ijm', A, B)
        return product
        
    def updater(self,old, new, alpha0 = 0.1):
        max_change = np.max(new/old)
        alpha  = np.min([alpha0,  alpha0/max_change])
        return old*(1-alpha) + new*alpha

    def c_s_updater(self, c_s_r_old, c_s_r_new, method='best', alpha_Picard = 0.1, alpha_oz = 0. ):
        if len(self.c_list)<3:
            method = 'fixed'
        if method=='best' or alpha_oz > 0.0:
            # c_s_r_oz =  self.FT_k_2_r_matrix(self.h_k_matrix  - self.γs_k_matrix)
            I_plus_h_rho_inverse = self.invert_matrix(self.I[:,:,np.newaxis] + self.h_k_matrix*self.rho[:,np.newaxis,np.newaxis])
            c_s_r_oz = self.FT_k_2_r_matrix(self.A_times_B(I_plus_h_rho_inverse, self.h_k_matrix))
        if method=='best':
            c_s_r = self.c_s_find_best_alpha(c_s_r_old, c_s_r_new, c_s_r_oz)
        elif method=='fixed':
            c_s_r = alpha_Picard*c_s_r_new + (1-alpha_Picard)*c_s_r_old
            if alpha_oz>0.0:
                c_s_r = alpha_oz*c_s_r_oz  + (1-alpha_oz)*c_s_r

        return c_s_r

    def c_s_find_best_alpha(self, c_s_r_old, c_s_r_new, c_s_r_oz):
        c_s_list = self.c_list + self.βu_l_k_matrix
        def c_err_1(αs):
            c_s = (1-sum(αs))*c_s_r_new + αs[0]*c_s_list[-1] + αs[1]*c_s_list[-2] + αs[2]*c_s_list[-3]
            err = self.total_err(c_s)
            # print("new_err: ", self.total_err(c_s_r_new))
            print("HNC αs: {0}, c_err: {1:.5e} ".format(np.array(αs), err))
            return err

        α_bounds = (0, 1)
        
        method='L-BFGS-B'
        method='SLSQP'
        method='Nelder-Mead'
        tol=1e-6
        res = minimize(c_err_1, [0.9,0.01,0.01], bounds = [(0,1),(0,1),(0,1)], tol=tol, options={'maxiter':int(1e2)}, method=method) #, constraints=constraints, method='COBYLA')#bounds=[(ε, 1-ε),(ε, 1-ε),(ε, 1-ε)]
        αs = res.x
        print(" HNC min:", res.x, res.success, res.message)
        c_s_r = (1-sum(αs))*c_s_r_new + αs[0]*c_s_list[-1] + αs[1]*c_s_list[-2] + αs[2]*c_s_list[-3]
        return  c_s_r 

    def total_err(self, c_s_r_matrix):
        c_s_k_matrix = self.FT_r_2_k_matrix(c_s_r_matrix)
        c_r_matrix = c_s_r_matrix - self.βu_l_r_matrix
        c_k_matrix = c_s_k_matrix - self.βu_l_k_matrix

        C_matrix = self.rho[np.newaxis,:,np.newaxis] * c_k_matrix
        γs_k_matrix = self.get_γs_k_matrix(c_s_k_matrix)
        h_k_matrix = c_s_k_matrix  + γs_k_matrix
        h_r_matrix = self.FT_k_2_r_matrix(h_k_matrix)

        tot_eqn =  1 + h_r_matrix  - np.exp(-self.βu_s_r_matrix + h_r_matrix - c_s_r_matrix )
        
        tot_err = np.linalg.norm(tot_eqn) /np.sqrt(self.N_bins*self.N_species**2)

        #print("tot: {0:.2e} ".format(tot_err))

        return tot_err
    
    

    def excess_energy_density(self):

        u_matrix = self.βu_r_matrix*self.Temp_matrix[:,:,np.newaxis]
        g_matrix = self.h_r_matrix+1
        rho_matrix = self.rho[:,np.newaxis]*self.rho[np.newaxis,:]
        r = self.r_array[np.newaxis,np.newaxis,:]
        dr = self.del_r
        potential_species_sum = np.sum(rho_matrix[:,:,np.newaxis]*u_matrix*g_matrix*r**2*dr)

        u_ex = 2*π*potential_species_sum

        return u_ex

    def get_new_c_k_matrix(self, c_s_k_matrix):
        γs_k_matrix = self.get_γs_k_matrix(c_s_k_matrix)                           # 1. c_k, u_l_k -> γ_k   (Definition)

        γs_r_matrix = self.FT_k_2_r_matrix(γs_k_matrix) # γ_k        -> γ_r   (FT)     
        h_r_matrix = -1 + np.exp(γs_r_matrix - βu_s_r_matrix) # 2. γ_r,u_s_r  -> h_r   (HNC)   
        h_r_matrix = np.where(h_r_matrix>h_max, h_max, h_r_matrix)
        h_k_matrix = self.FT_r_2_k_matrix(h_r_matrix)
        # Plug into HNC equation

        new_c_s_r_matrix = h_r_matrix - γs_r_matrix # 3. h_r, γ_r   -> c_s_r (Ornstein-Zernicke)
        new_c_s_k_matrix = self.FT_r_2_k_matrix(new_c_s_r_matrix)
        return new_c_s_k_matrix

    def HNC_GMRES_solve(self, h_max=200, alpha_method='best'):
        gmres_matrix_size = self.N_species**2*self.N_bins

        def A_matrix(c_s_k_flat):
            c_s_k_matrix = c_s_k_flat.reshape(self.N_species, self.N_species, self.N_bins)
            return self.get_new_c_k_matrix(c_s_k_matrix)

        


        A = LinearOperator((gmres_matrix_size, gmres_matrix_size ), matvec= A_matrix)

        iters_per_cycle = 100#int(self.grid.Ntot/2)
        number_cycles   = 100
        gmres_kwargs={'tol':1e-3,'restart': iters_per_cycle ,'maxiter':number_cycles,'callback':caller,
          'callback_type':'pr_norm'}#, 'M':M_preconditioner}
        x0 = 
        c_s_k_sol_flat  , self.info = gmres(A, b, x0 = x0, **gmres_kwargs)


    # Solver
    def HNC_solve(self, h_max=200, alpha_method='best', alpha_Picard = 0.1, alpha_oz = 0., iters_to_check=10 ):
        """ 
        Integral equation solutions for the classical electron gas 
        J. F. Springer; M. A. Pokrant; F. A. Stevens, Jr.
        Crossmark: Check for Updates
        J. Chem. Phys. 58, 4863–4867 (1973)
        https://doi.org/10.1063/1.1679070

        Their N is my γ = h - c
        1. c_k, u_l_k -> γ_k   (Definition)
        2. γ_r,u_s_r  -> h_r   (HNC)
        3. h_r, γ_r   -> c_s_r (Ornstein-Zernicke)
        4. c_s, u_l   -> c_r_k (Definition)
        Args: 
            species_left: tuple specifying which species to solve for
        Returns:
            None
        """

        converged = 1
        decreasing = True
        iteration = 0
        self.tot_err_list = []
        self.h_list, self.c_list, self.c_k_list = [], [], []
        self.u_ex_list = []
        self.h_list.append(self.h_r_matrix.copy())            
        self.c_list.append(self.c_r_matrix.copy())
        self.c_k_list.append(self.c_k_matrix.copy())
        while converged!=0:
            # Compute matrices in k-space using OZ equation
            # if iteration>=0:
            self.set_γs_k_matrix()                           # 1. c_k, u_l_k -> γ_k   (Definition)

            self.γs_r_matrix = self.FT_k_2_r_matrix(self.γs_k_matrix) # γ_k        -> γ_r   (FT)     
            self.βω_r_matrix = self.βu_s_r_matrix - self.γs_r_matrix   # potential of mean force
            self.h_r_matrix = -1 + np.exp(self.γs_r_matrix - self.βu_s_r_matrix) # 2. γ_r,u_s_r  -> h_r   (HNC)   
            self.h_r_matrix = np.where(self.h_r_matrix>h_max, h_max, self.h_r_matrix)
            self.h_k_matrix = self.FT_r_2_k_matrix(self.h_r_matrix)
            # Plug into HNC equation

            new_c_s_r_matrix = self.h_r_matrix - self.γs_r_matrix # 3. h_r, γ_r   -> c_s_r (Ornstein-Zernicke)
            
            # Update h_r, c_r_matrix
            old_c_s_r_matrix = self.c_s_r_matrix.copy()
            self.c_s_r_matrix = self.c_s_updater(old_c_s_r_matrix, new_c_s_r_matrix, method = alpha_method, alpha_Picard = alpha_Picard, alpha_oz = alpha_oz )
            
            self.c_s_k_matrix = self.FT_r_2_k_matrix(self.c_s_r_matrix)  # FT
            
            self.c_r_matrix = self.c_s_r_matrix  - self.βu_l_r_matrix # 4. c_s, u_l   -> c_r_k (Definition)
            self.c_k_matrix = self.c_s_k_matrix  - self.βu_l_k_matrix# FT
            self.set_C_matrix()  # Update C = rho c    

            self.h_list.append(self.h_r_matrix.copy())            
            self.c_list.append(self.c_r_matrix.copy())
            self.c_k_list.append(self.c_k_matrix.copy())

            # oz_err = np.linalg.norm(-self.h_k_matrix + self.c_k_matrix  + self.A_times_B(self.c_k_matrix, self.h_k_matrix*self.rho[:,np.newaxis,np.newaxis]))/np.sqrt(self.N_bins*self.N_species**2)
            oz_err = np.linalg.norm(-self.h_k_matrix + self.c_s_k_matrix  + self.γs_k_matrix)/np.sqrt(self.N_bins*self.N_species**2)
            hnc_err = np.linalg.norm(- 1 - self.h_r_matrix   + np.exp( -self.βu_r_matrix + self.h_r_matrix - self.c_r_matrix ))/np.sqrt(self.N_bins*self.N_species**2)
                        # Compute change over iteration

            tot_err = np.linalg.norm(- 1 - self.c_s_r_matrix  - self.γs_r_matrix   + np.exp( -self.βu_s_r_matrix  + self.γs_r_matrix  ))/np.sqrt(self.N_bins*self.N_species**2)
            err_c = np.linalg.norm(old_c_s_r_matrix - self.c_s_r_matrix) / np.sqrt(self.N_bins*self.N_species**2)
            u_ex = self.excess_energy_density()
            self.u_ex_list.append(u_ex)

            actual_tot_err = self.total_err(self.c_s_r_matrix)
            self.tot_err_list.append(actual_tot_err)
            if iteration%50==0:
                print("{0}: Err in c_r: {1:.2e}, OZ: {2:.2e}, HNC: {3:.2e}, tot: {4:.2e}, tot: {5:.2e}".format(iteration,err_c, oz_err, hnc_err, tot_err, actual_tot_err))
            
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

            if isnan(err_c):
                print("QUIT: c_r is nan.")
                break
            if iteration > self.num_iterations:
                converged=1
                print("Warning: HNC_solver did not converge within the specified number of iterations.")                
                break

            if err_c < self.tol:
                converged = 0

            iteration += 1
            
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

    def invert_HNC(self, removed_species):
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

    ############# PLOTTING #############
    def plot_species(self, species_nums):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16,10))
        self.c_r_matrix = self.FT_k_2_r_matrix(self.c_k_matrix)
        fig.suptitle("Species: " + self.name_matrix[species_nums[0]][species_nums[1]] ,fontsize=20)
        # Top Left
        axs[0,0].plot(self.r_array, self.c_r_matrix[species_nums],'--.')
        axs[0,0].set_xlabel("r",fontsize=20)
        axs[0,0].set_ylabel("c(r)",fontsize=20)
        axs[0,0].set_title("Direct Correlation function",fontsize=20)
        axs[0,0].set_yscale('symlog',linthresh=0.1)
        
        # Top Right
        axs[0,1].plot(self.k_array, self.c_k_matrix[species_nums],'--.')
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
        axs[1,1].plot(self.r_array, np.exp(self.γs_r_matrix + self.βu_l_r_matrix)[species_nums] ,'--.')
        axs[1,1].set_xlabel("r/r_s",fontsize=20)
        axs[1,1].set_ylabel(r"$e^\gamma(r)$",fontsize=20)
        axs[1,1].set_title(r"$n=n_{ideal}(r) e^{\gamma(r)}$",fontsize=20)
        axs[1,1].set_yscale('symlog',linthresh=0.1)

        for ax in axs.flatten():
            ax.tick_params(labelsize=20)
            ax.set_xscale('log')

    

        plt.tight_layout()
        # plt.show()

    def plot_convergence_uex(self):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
        ax.plot(self.u_ex_list)
        ax.set_title( "Excess Potential Energy Density" ,fontsize=15)
        # ax.set_ylim(0, np.max(  [np.max(self.h_list[k][i,j]+1), 5]) )
        # ax.set_xlim(0,41)
        ax.tick_params(labelsize=20)
        ax.set_xlabel("Iteration",fontsize=20)
        
        ax.set_ylabel(r"$u_{\rm ex} r_s^3$ [A.U.]",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        plt.show()

    def plot_species_convergence_g(self, n_slices=4):
        fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))
        fig.suptitle("Radial Distribution Convergence Blue to Red" ,fontsize=20)
        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.r_array, self.h_list[k][i,j]+1, color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Gamma[i][j]) ,fontsize=15)
                    axs[i,j].set_ylim(0, np.max(  [np.max(self.h_list[k][i,j]+1), 5]) )
                    axs[i,j].set_xlim(0,4)
                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$r/r_s$",fontsize=20)
                    if np.max(self.h_list[k][i,j]+1) > 5:
                        axs[i,j].set_yscale('symlog',linthresh=10)

            axs[i,0].set_ylabel(r"$g(r/r_s)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)

        plt.show()

    def plot_species_convergence_c(self, n_slices=4):
        fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))
        fig.suptitle("Direct Correlation Function Convergence Blue to Red" ,fontsize=20)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.r_array, self.c_list[k][i,j], color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Gamma[i][j]) ,fontsize=15)
                    axs[i,j].set_xlim(0,4)

                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$r/r_s$",fontsize=20)
                    axs[i,j].set_yscale('symlog',linthresh=0.1)

            axs[i,0].set_ylabel(r"$c(r/r_s)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        plt.show()

    def plot_species_convergence_ck(self, n_slices=4):
        fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(10*self.N_species,6*self.N_species))
        fig.suptitle("Direct Correlation Function Convergence Blue to Red" ,fontsize=20)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        n = len(self.h_list)
        indcs = [int(num)-1 for num in n/n_slices*np.arange(1,n_slices+1)]

        for i in range(self.N_species):
            for j in range(self.N_species):
                for k in indcs:
                    color = plt.cm.jet(k/n)
                    axs[i,j].plot(self.k_array, self.c_k_list[k][i,j], color=color ,label='iter: {0}'.format(int(k)))
                    axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Gamma[i][j]) ,fontsize=15)
                    # axs[i,j].set_xlim(0,10)
                    axs[i,j].set_xscale('log')

                    axs[i,j].tick_params(labelsize=20)
                    axs[-1,j].set_xlabel(r"$k $",fontsize=20)
                    axs[i,j].set_yscale('symlog',linthresh=0.1)

            axs[i,0].set_ylabel(r"$c(k)$",fontsize=20)

        #axs[0,0].legend(fontsize=20)
        
        plt.show()

    def plot_g_grid(self):
        fig, axs = plt.subplots(ncols=self.N_species, nrows=self.N_species, figsize=(8*self.N_species,4*self.N_species))
        fig.suptitle("Radial Distribution Function for all Species",fontsize=20,y=1)

        if type(axs) not in [list, np.ndarray, tuple]: 
            axs=np.array([[axs]])

        for i in range(self.N_species):
            for j in range(self.N_species):
                axs[i,j].plot(self.r_array, self.h_r_matrix[i,j]+1)
                axs[i,j].set_title(self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Gamma[i][j]) ,fontsize=15)
                axs[i,j].set_ylim(0,2)
                axs[i,j].tick_params(labelsize=20)
                axs[-1,j].set_xlabel(r"$r/r_s$",fontsize=20)

            axs[i,0].set_ylabel(r"$g(r/r_s)$",fontsize=20)
                

        for ax in axs.flatten():
            ax.tick_params(labelsize=15)
        
        plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95], pad=0.4, w_pad=5.0, h_pad=5.0)
        plt.show()

    def plot_g_all_species(self, data_to_compare=None, data_names=None, gmax=None):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8*self.N_species,4*self.N_species))
        fig.suptitle("Radial Distribution Function for all Species",fontsize=20,y=1)

        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.r_array, self.h_r_matrix[i,j]+1,'--.', label=self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Gamma[i][j]) )
        
        if data_to_compare==None:
            pass
        else:
            for file_name, label in zip(data_to_compare, data_names):
                r_datas, g_datas = np.array(read_csv(file_name,delimiter=',',header=1)).T
                ax.plot(r_datas, g_datas,'-', label=label)
                ax.legend(fontsize=15)

        ax.plot()


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
        plt.show()

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
        plt.show()

    def plot_βu_all_species(self):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,6))
        fig.suptitle("Potential Energy between all Species",fontsize=20,y=1)

        for i in range(self.N_species):
            for j in range(self.N_species):
                ax.plot(self.r_array, self.βu_r_matrix[i,j], label=self.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j,self.Gamma[i][j]) )
        
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
        plt.show()

if __name__ == "__main__":
    N_species = 3
    Gamma = np.array(  [[1,  0.1,  3],
                        [0.1,  5,   3],
                        [3,    3,   5]])
    names = ["Ion-1","Ion-2", "Electron", ] 
    kappa = 3
    rho = np.array([  3/(4*np.pi), 3/(4*np.pi), 3/(4*np.pi)  ])
    hnc = HNC_solver(N_species, Gamma=Gamma, kappa=kappa, tol=3e-1, alpha=1e-3, rho = rho, num_iterations=int(1e4), R_max=10, N_bins=1000, names=names)

    self.HNC_solve()
    # print(self.c_r_matrix)

    h_r_matrix = self.h_r_matrix
    c_r_matrix = self.c_r_matrix

    # To get the radial distribution function g(r), you can add 1 to the h_r_matrix:
    g_r_matrix = h_r_matrix + 1

    # To visualize the results, you can use matplotlib to plot the g(r) function:
    # self.plot_species((0,0))
    # self.plot_species_convergence((0,0))
    self.plot_g_all_species()
    
   

