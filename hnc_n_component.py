import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt


class HNC_solver():
    def __init__(self, N_species, Gamma, rho, kappa = 1.0, alpha=1.0, tol=1e-4, num_iterations=1000, R_max=25.0, N_bins=50, names=None):
        self.N_species = N_species
        self.Gamma = Gamma
        self.rho = rho
        self.kappa = kappa
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.tol = tol
        self.R_max = R_max
        self.N_bins = N_bins

        self.I = np.eye(N_species)
        self.i = slice(None,self.N_species)
        self.j = slice(None,self.N_species)

        self.make_k_r_spaces()

        self.initialize()
        if names!=None:
            self.names=names
            self.make_name_matrix()

    def make_name_matrix(self):
        self.name_matrix = [[f'{a}-{b}' if a != b else a for b in self.names] for a in self.names]


    def initialize(self):
        self.initialize_βu_matrix()
        self.initialize_c_k()
        self.initialize_C_matrix()
        self.get_S_k_matrix()
        self.get_h_k_matrix()
        self.h_r_matrix = self.get_h_r_matrix()
        self.get_c_r_matrix()

    def make_k_r_spaces(self):
        self.del_r = self.R_max / self.N_bins
        self.r_array = np.linspace(self.del_r / 2, self.R_max - self.del_r / 2, self.N_bins)

        self.del_k = np.pi / ((self.N_bins + 1) * self.del_r)
        self.K_max = self.del_k * self.N_bins
        self.k_array = np.linspace(self.del_k / 2, self.K_max - self.del_k / 2, self.N_bins)

        self.fact_r_2_k = 2 * np.pi * self.del_r
        self.fact_k_2_r = self.del_k / (4. * np.pi**2)
        self.dimless_dens = 3. / (4 * np.pi)

    def FT_r_2_k(self, input_array):
        from_dst = self.fact_r_2_k * fftpack.dst(self.r_array * input_array)
        return from_dst / self.k_array

    def FT_k_2_r(self, input_array):
        from_idst = self.fact_k_2_r * fftpack.idst(self.k_array * input_array)
        return from_idst / self.r_array

    def FT_r_2_k_matrix(self, f_r):
        i, j = self.i, self.j
        N = f_r.shape[0]
        f_k = np.zeros((N, N, self.N_bins))
        f_k[i,j] = self.FT_r_2_k(f_r[i,j])
        return f_k

    def FT_k_2_r_matrix(self, f_k):
        i, j = self.i, self.j
        N = f_k.shape[0]
        f_r = np.zeros((N, N, self.N_bins))
        f_r[i,j] = self.FT_k_2_r(f_k[i,j])
        return f_r

    def initialize_βu_matrix(self):
        i, j = self.i, self.j
        # r = slice(0,len(self.r_))
        self.βu_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        self.βu_matrix[i, j] = (np.exp(- self.kappa * self.r_array) / self.r_array)
        self.βu_matrix = self.βu_matrix * self.Gamma[:,:,np.newaxis]

    def initialize_c_k(self):
        i, j = self.i, self.j
        c_k = np.zeros((self.N_species, self.N_species, self.N_bins))
        c_k[i, j] = -4 * np.pi * self.Gamma[i, j, np.newaxis] / (self.k_array**2 + self.kappa**2)[np.newaxis,np.newaxis,:]
        self.c_k_matrix = c_k

    def initialize_C_matrix(self):
        """
        Defines matrix of rho_i c_ij using initial assumption of diagonal
        """
        i, j = self.i, self.j
        self.C_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        self.C_matrix[i, j] = self.rho[i,np.newaxis,np.newaxis] * self.c_k_matrix[i, j]
    
    def get_C_matrix(self):
        """
        Defines matrix of rho_i c_ij using initial assumption of diagonal
        """
        i, j = self.i, self.j
        self.C_matrix[i, j] = self.rho[i,np.newaxis,np.newaxis] * self.c_k_matrix[i, j]


    def set_C_matrix(self):
        """
        Defines matrix of rho_i c_ij using initial assumption of diagonal
        """
        i, j = self.i, self.j
        self.C_matrix[i, j] = self.rho[i,np.newaxis,np.newaxis] * self.c_k[i, j]

    def invert_matrix(self, A_matrix):
        """
        Calls routine to invert an nxn matrix
        """
        A_inverse = np.linalg.inv(A_matrix)
        return A_inverse

    def get_S_k_matrix(self):
        """
        invert N_species x N_species  matrix equation to get S_k
        """
        i, j = self.i, self.j

        self.S_k_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        for k_index in range(self.N_bins):
            self.S_k_matrix[i, j, k_index] = self.invert_matrix(self.I[i,j] - self.C_matrix[i,j, k_index])

    def get_h_k_matrix(self):
        """
        Calculate h_k_matrix using S_k_matrix and rho
        """
        i, j = self.i, self.j
        self.h_k_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        self.h_k_matrix[i, j] = (self.S_k_matrix[i, j] - self.I[i, j, np.newaxis]) / self.rho[i, np.newaxis,np.newaxis]

    def get_h_r_matrix(self):
        """
        Fourier transform h_k to r space
        """
        i, j = self.i, self.j
        new_h_r_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        new_h_r_matrix[i, j] = self.FT_k_2_r(self.h_k_matrix[i, j])

        new_h_r_matrix[i, j] = np.where(new_h_r_matrix[i, j]<-1, -1+1e-5, new_h_r_matrix[i, j])
        return new_h_r_matrix

    def get_c_r_matrix(self):
        """
        Fourier transform C_matrix to c in r space
        """
        i, j = self.i, self.j
        self.c_r_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        self.c_r_matrix[i, j] = self.FT_k_2_r(self.C_matrix[i, j]) / self.rho[i,np.newaxis,np.newaxis]

    def get_lnγ_r_matrix(self, h_r_matrix):
        i, j = self.i, self.j
        lnγ_r_matrix = np.zeros((self.N_species, self.N_species, self.N_bins))
        # print("1+h", 1 + h_r_matrix[i, j])
        lnγ_r_matrix[i, j] = np.log(1 + h_r_matrix[i, j]) + self.βu_matrix[i, j]

        return lnγ_r_matrix

    def HNC_update_c(self, h_r_matrix):
        """
        Runs HNC with last iteration h_ij, c_ij and u_ij to get new h_ij
        """
        i, j = self.i, self.j
        lnγ_r_matrix = self.get_lnγ_r_matrix(h_r_matrix)
        c_r = np.zeros((self.N_species, self.N_species, self.N_bins))
        c_r[i, j] = h_r_matrix[i, j] - lnγ_r_matrix[i, j]
        return c_r

    def HNC_solve(self, alpha=1e-2):

        converged = False
        iteration = 0
        self.h_list = []
        while not converged and iteration < self.num_iterations:
            # Compute matrices in k-space using OZ equation
            self.get_S_k_matrix() # Takes in C matrix, computes S
            self.get_h_k_matrix() # Gets h_k from S

            # Fourier h_k and cutoff so non-negative
            self.h_r_matrix = self.FT_k_2_r_matrix(self.h_k_matrix) # New h_r from FT of h_k 
            soft_ε = 1e-6*(np.exp(self.r_array)-1)
            self.h_r_matrix = np.where(self.h_r_matrix[self.i, self.j]<-1, -1+soft_ε, self.h_r_matrix[self.i, self.j])

            # Plug into HNC equation
            self.c_r_matrix = self.HNC_update_c(self.h_r_matrix) # compute γ = (1+h) e^βu, c = h-logγ
            new_c_k_matrix = self.FT_r_2_k_matrix(self.c_r_matrix)

            # Compute change over iteration
            err_c = np.linalg.norm(new_c_k_matrix - self.c_k_matrix) / np.sqrt(self.N_bins*self.N_species)
            # err_h = np.linalg.norm(new_h_r_matrix - self.h_r_matrix) / np.sqrt(self.N_bins*self.N_species)

            # Update h_r, c_r_matrix
            self.c_k_matrix = self.c_k_matrix*(1-self.alpha) + self.alpha*new_c_k_matrix
            self.get_C_matrix()     

            self.h_list.append(list(self.h_r_matrix[0,0,:]))
            # print(self.h_r_matrix[0,0,:])
            
            if self.num_iterations%100==0:
                print("Err in c_r: {0:.3f}".format(err_c))
            # print("Err in h_r: {0:.3f}".format(err_h))

            if err_c < self.tol:
                converged = True


            iteration += 1

        if not converged:
            print("Warning: HNC_solver did not converge within the specified number of iterations.")

    @staticmethod
    def remove_species(matrix, species):
        """
        Removes a specific species column and row from arrays
        """
        matrix = np.delete(matrix, species, axis=0)
        matrix = np.delete(matrix, species, axis=1)
        return matrix

    def initialize_effective(self):
        self.Seff_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.Ceff_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.ceff_k_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.ceff_r_matrix = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        self.βueff = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))

    def invert_HNC(self, removed_species, alpha=1e-2, tol=1e-3):
        """ 
        Takes N species results and inverts it to find single effective v_ii for one species
        1. h_k -> S   (Definition)
        2. S   -> C   (Ornstein-Zernicke)
        3. C   -> c_k   (definition)
        4. c_k -> c_r    (FT)
        5. h_r, c_r -> βu_r (HNC)
        Args: 
            species_left: tuple specifying which species to solve for
        Returns:
            None
        """
        self.Neff_species = self.N_species - 1
        # First get new effective h matrix
        self.rhoeff        = np.delete(self.rho, removed_species)
        self.heff_r_matrix = self.remove_species(self.h_r_matrix, removed_species)
        self.heff_k_matrix = self.FT_r_2_k_matrix(self.heff_r_matrix)
            
        #Initialize other matrices to new size
        self.initialize_effective()

        # Now Inverting steps
        converged=False
        ieff, jeff = slice(None, self.Neff_species), slice(None,self.Neff_species)
        new_βueff = np.zeros((self.Neff_species,self.Neff_species,self.N_bins))
        iteration=0
        while not converged and iteration < self.num_iterations:
            self.Seff_matrix[ieff, jeff] = self.I[ieff, jeff, np.newaxis]  + self.heff_r_matrix * self.rhoeff[ieff, np.newaxis,np.newaxis]   #   1. h_k -> S (Ornstein-Zernicke)
            for k_index in range(self.N_bins):
                self.Ceff_matrix[ieff, jeff,k_index]  = self.I[ieff, jeff] - self.invert_matrix(self.Seff_matrix[ieff, jeff,k_index]) #   2. S -> C 
            self.ceff_k_matrix[ieff, jeff] = self.Ceff_matrix[ieff,jeff]/self.rho[ieff,np.newaxis,np.newaxis] # 3. C   -> c_k
            self.ceff_r_matrix[ieff, jeff] = self.FT_k_2_r(self.ceff_k_matrix[ieff,jeff]) #4. c_k -> c_r
            new_βueff[ieff,jeff]   = self.heff_r_matrix[ieff,jeff] + self.ceff_r_matrix[ieff,jeff] - np.log(1+self.heff_r_matrix[ieff,jeff]) #  h_r, c_r -> βu_r

            #Compute change/err
            err = np.linalg.norm(self.βueff - new_βueff)/np.sqrt(self.Neff_species*self.N_bins)
            if iteration%100==0:
                print("βu Err: {0:.3e}".format(err))
            self.βueff = self.βueff*(1-self.alpha) + self.alpha*new_βueff            
            if err < tol:
                converged=True
            iteration+=1
        if not converged:
            print("Warning: HNC_solver did not converge within the specified number of iterations.")


    def plot_species(self, species_nums):
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20,10))
        fig.suptitle("Species: " + self.name_matrix[species_nums[0]][species_nums[1]] ,fontsize=20)
        # Top Left
        axs[0,0].plot(self.r_array, c_r_matrix[species_nums])
        axs[0,0].set_xlabel("r",fontsize=15)
        axs[0,0].set_ylabel("c(r)",fontsize=15)
        axs[0,0].set_title("Direct Correlation function",fontsize=15)
        # axs[0,0].set_ylim(-1,2)
        
        # Top Right
        axs[0,1].plot(self.k_array, self.c_k_matrix[species_nums])
        axs[0,1].set_xlabel("k",fontsize=15)
        axs[0,1].set_ylabel("c(k)",fontsize=15)
        axs[0,1].set_title("Fourier Direct Correlation function",fontsize=15)
        # axs[0,1].set_ylim(-1,2)

        # Bottom Left
        axs[1,0].plot(self.r_array, g_r_matrix[species_nums])
        axs[1,0].set_xlabel("r",fontsize=15)
        axs[1,0].set_ylabel("g(r)",fontsize=15)
        axs[1,0].set_title("Radial distribution function",fontsize=15)
        axs[1,0].set_ylim(-1,2)
        
        # Bottom Right
        axs[1,1].plot(self.r_array, self.S_k_matrix[species_nums])
        axs[1,1].set_xlabel("k",fontsize=15)
        axs[1,1].set_ylabel("S(k)",fontsize=15)
        axs[1,1].set_title("Static structure factor",fontsize=15)

        for ax in axs.flatten():
            ax.tick_params(labelsize=15)
        
        plt.tight_layout()
        # plt.show()

    def plot_species_convergence(self,species_nums):
        fig, ax = plt.subplots()
        fig.suptitle("Species: " + self.name_matrix[species_nums[0]][species_nums[1]] ,fontsize=20)

        n = len(hnc.h_list)
        indcs = [int(num)-1 for num in n/4*np.arange(1,5)]

        for i in indcs:
            ax.plot(hnc.r_array, hnc.h_list[i], label='iteration {0}'.format(int(i)))
        
        ax.set_ylabel(r'$h(r/r_s)$')
        ax.set_xlabel(r"$r/r_s$")
        ax.legend()
        plt.show()

    def plot_g_all_species(self):
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

if __name__ == "__main__":
    N_species = 3
    Gamma = np.array(  [[1,  0.1,  3],
                        [0.1,  5,   3],
                        [3,    3,   5]])
    names = ["Ion-1","Ion-2", "Electron", ] 
    kappa = 3
    rho = np.array([  3/(4*np.pi), 3/(4*np.pi), 3/(4*np.pi)  ])
    hnc = HNC_solver(N_species, Gamma=Gamma, kappa=kappa, tol=3e-1, alpha=1e-3, rho = rho, num_iterations=int(1e4), R_max=10, N_bins=1000, names=names)

    hnc.HNC_solve()
    # print(hnc.c_r_matrix)

    h_r_matrix = hnc.h_r_matrix
    c_r_matrix = hnc.c_r_matrix

    # To get the radial distribution function g(r), you can add 1 to the h_r_matrix:
    g_r_matrix = h_r_matrix + 1

    # To visualize the results, you can use matplotlib to plot the g(r) function:
    # hnc.plot_species((0,0))
    # hnc.plot_species_convergence((0,0))
    hnc.plot_g_all_species()
    
   

