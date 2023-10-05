# Python file for running hnc TCCW 2 cases
# Zach Johnson
# June 2023

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from time import time

# from hnc_custom_screeningOZ_multiscale import  HNC_solver
# from hnc import  HNC_solver
from hnc_Ng import  HNC_solver
from qsps import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
def set_hnc_onlyion(n_in_per_cc, T, Z, A, Zbar, c_k_guess=None , which_Tij='thermal',
            oz_type='standard', method='kryov', add_bridge=False, bridge='yukawa',
            pseudopotential=False, r_c=0.6, no_coupling=False, N_bins = 500, R_max=5):
    
    n_in_AU = n_in_per_cc*1e6 *aB**3
    ri = QSP_HNC.rs_from_n(n_in_AU)
    qsp = QSP_HNC(Z, A, Zbar, T, T, ri, Zbar*n_in_AU, which_Tij=which_Tij, r_c=r_c)

    N_species = 1
    Gamma = np.array(  [[qsp.Γii]])


    names = ["Ion-1", "Electron", ] 
    kappa = 1
    
    rhos = np.array([  3/(4*np.pi)])
    temps = np.array([qsp.Ti])
    masses= np.array([qsp.m_i])
    hnc1 = HNC_solver(N_species, Gamma, rhos, temps, masses ,
                     kappa_multiscale=5, R_max=R_max, N_bins=N_bins, 
                      names=names, dst_type=3, oz_method=oz_type)

    if pseudopotential==True:
        βvei = qsp.βvei_atomic(hnc1.r_array)
    else:
        βvei = qsp.βvei(hnc1.r_array)
    βu_r_matrix = np.array([[qsp.βvii(hnc1.r_array)]])
    if no_coupling:
        βu_r_matrix[0,1]=0
        βu_r_matrix[1,0]=0
    
    if add_bridge:
        if bridge=='ocp':
            βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_OCP(hnc1.r_array, qsp.Γii)
        elif bridge=='yukawa':
            βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_Yukawa(hnc1.r_array, qsp.Γii, qsp.get_κ())
    hnc1.set_βu_matrix(βu_r_matrix)
    hnc1.initialize_c_k()
#     print("c asymm: ", hnc1.c_s_k_matrix[1,0]/hnc1.c_s_k_matrix[0,1])
    if c_k_guess is not None:
        for i in range(N_species):
            for j in range(N_species):
                if (c_k_guess[i,j]!=np.zeros(hnc1.N_bins)).all():
#                 if i==j:
                    hnc1.c_k_matrix[i,j] = c_k_guess[i,j]
                    hnc1.c_r_matrix[i,j] = hnc1.FT_k_2_r(hnc1.c_k_matrix[i,j])
                    hnc1.c_s_k_matrix[i,j] = hnc1.c_k_matrix[i,j] + hnc1.βu_l_k_matrix[i,j]
                    hnc1.c_s_r_matrix[i,j] = hnc1.c_r_matrix[i,j] + hnc1.βu_l_r_matrix[i,j]

#     print("c asymm: ", hnc1.c_s_k_matrix[1,0]/hnc1.c_s_k_matrix[0,1])
    hnc1.set_C_matrix()

    return hnc1, qsp
def set_hnc(n_in_per_cc, T, Z, A, Zbar, c_k_guess=None , which_Tij='thermal',
            oz_type='standard', method='kryov', add_bridge=False, bridge='yukawa',
            pseudopotential=False, r_c=0.6, no_coupling=False, N_bins = 500, R_max=5):
    
    n_in_AU = n_in_per_cc*1e6 *aB**3
    ri = QSP_HNC.rs_from_n(n_in_AU)
    qsp = QSP_HNC(Z, A, Zbar, T, T, ri, Zbar*n_in_AU, which_Tij=which_Tij, r_c=r_c)

    N_species = 2
    Gamma = np.array(  [[qsp.Γii,  qsp.Γei],
                        [qsp.Γei,  qsp.Γee]])


    names = ["Ion-1", "Electron", ] 
    kappa = 1
    
    rhos = np.array([  3/(4*np.pi), Zbar*3/(4*np.pi) ])
    temps = np.array([qsp.Ti, qsp.Te_c])
    masses= np.array([qsp.m_i, m_e])
    hnc1 = HNC_solver(N_species, Gamma, rhos, temps, masses ,
                     kappa_multiscale=5, R_max=R_max, N_bins=N_bins, 
                      names=names, dst_type=3, oz_method=oz_type)

    if pseudopotential==True:
        βvei = qsp.βvei_atomic(hnc1.r_array)
    else:
        βvei = qsp.βvei(hnc1.r_array)
    βu_r_matrix = np.array([[qsp.βvii(hnc1.r_array), βvei],
                            [βvei, qsp.βvee(hnc1.r_array)]])
    if no_coupling:
        βu_r_matrix[0,1]=0
        βu_r_matrix[1,0]=0
    
    if add_bridge:
        if bridge=='ocp':
            βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_OCP(hnc1.r_array, qsp.Γii)
        elif bridge=='yukawa':
            βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_Yukawa(hnc1.r_array, qsp.Γii, qsp.get_κ())
    hnc1.set_βu_matrix(βu_r_matrix)
    hnc1.initialize_c_k()
#     print("c asymm: ", hnc1.c_s_k_matrix[1,0]/hnc1.c_s_k_matrix[0,1])
    if c_k_guess is not None:
        for i in range(N_species):
            for j in range(N_species):
                if (c_k_guess[i,j]!=np.zeros(hnc1.N_bins)).all():
#                 if i==j:
                    hnc1.c_k_matrix[i,j] = c_k_guess[i,j]
                    hnc1.c_r_matrix[i,j] = hnc1.FT_k_2_r(hnc1.c_k_matrix[i,j])
                    hnc1.c_s_k_matrix[i,j] = hnc1.c_k_matrix[i,j] + hnc1.βu_l_k_matrix[i,j]
                    hnc1.c_s_r_matrix[i,j] = hnc1.c_r_matrix[i,j] + hnc1.βu_l_r_matrix[i,j]

#     print("c asymm: ", hnc1.c_s_k_matrix[1,0]/hnc1.c_s_k_matrix[0,1])
    hnc1.set_C_matrix()

    return hnc1, qsp

from scipy.optimize import curve_fit

class βu_fit():
    def __init__(self, func, r_array, y_data, initial_guess):
        self.r = r_array
        self.y = y_data
        self.y_vals = curve_fit(func, r_array, y_data, maxfev=int(1e5), p0=initial_guess)
        self.y_fit  = func(r_array, *self.y_vals[0])
        self.err = np.linalg.norm(self.y_fit-self.y)
        print(func.__name__ + " error: {0:.3e} ".format(self.err))

def yukawa_plus(r, a, b, c, d):
    return  a/r*np.exp(-b*r)/(1+np.exp(c*(r-d)))

def yukawa_plus_gaussian(r, a ,b ,c, d ,e, f, g):
    return  a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g)

def yukawa_plus_cos(r, a ,b ,c, d , h, i, j, k, l):
    return  a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + h*np.cos((r-i)*j*np.exp(-k*r))*np.exp(-l*r)

def yukawa_plus_gaussian_cos(r, a ,b ,c, d ,e, f, g, h, i, j, k, l):
    return  a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.cos((r-f)*g*np.exp(-h*r))*np.exp(-i*r) + j*np.exp(-(k-r)**2/l)


#########################
#########################

# species_file = "/home/zach/plasma/hnc/data/TCCW_single_species_data.csv"
species_file = "/home/zach/plasma/hnc/data/TCCW_three_species_paper_data.csv"
tccw_species_data = read_csv(species_file)
tccw_cases = [tccw_species_data.iloc[n] for n in range(len(tccw_species_data))]

case_successes = {}
atom_case_successes = {}
max_attempts=1
t0 = time()
cases_converged_thusfar = {'H1':True,'C1':True, 'Al1':True, 'Cu1': True, 'Be1': True, 'Au1': False, 'H2': True, 'H3': True, 'C2': False, 'C3': True, 'Al2': False, 'Al3': True,
 'Cu2': False, 'Cu3': True, 'H11': False, 'H21': False, 'H31': False, 'C11': False, 'C21': False, 'C31': False, 'Al11': False, 'Al21': False, 
 'Al31': False, 'Cu11': False, 'Cu21': False, 'Cu31': False, 'H12': True, 'H22': True, 'H32': False, 'C12': True, 'C22': False, 'C32': False, 
 'Al12': True, 'Al22': False, 'Al32': False, 'Cu12': True, 'Cu22': False, 'Cu32': False, 'H13': True, 'H23': True, 'H33': True, 'C13': True,
  'C23': True, 'C33': False, 'Al13': False, 'Al23': False, 'Al33': False, 'Cu13': True, 'Cu23': False, 'Cu33': False, 'H14': False, 'H24': False, 
  'H34': False, 'C14': False, 'C24': False, 'C34': False, 'Al14': False, 'Al24': False, 'Al34': False, 'Cu14': False, 'Cu24': False, 'Cu34': False,
   'H15': False, 'H25': False, 'H35': False, 'C15': False, 'C25': False, 'C35': False, 'Al15': False, 'Al25': False, 'Al35': False, 'Cu15': False,
    'Cu25': False, 'Cu35': False, 'H16': False}

for tccw_case in [tccw_cases[0], tccw_cases[1], tccw_cases[2]]:
    t0=time()
    α = 0.1
    case_converged=False
    case_attempts=0
    case_num= tccw_case[' ']
    case_id = tccw_case['Case ID']
    ni = tccw_case['Number Density [N/cc]']
    Te = tccw_case['Temperature [eV]']*eV_to_AU
    Ti = Te
    Z = tccw_case['Atomic Number']
    Zbar = tccw_case['Zbar (AAM)']
    # Zbar = tccw_case['Zbar (TFDW)']
    A = tccw_case['Atomic Weight [a.u.]']
    r_s = tccw_case['Wigner-Seitz Radius [cm]']
    r_c = tccw_case['Average-Bound Radius [cm]']/r_s
    if r_c==0: r_c = 1e-4
    print(r_s, r_c)
    print('\n______________________________\nCase num: {0} Case ID: {1}'.format(case_num, case_id))
    print("Te = {0:.3e} eV, n_i = {1:.3e} 1/cc, r_c/r_s = {2:.3f}".format(Te/eV_to_AU, ni, r_c))
    # if cases_converged_thusfar[case_id]==True:
    #     print("Already converged, skipping.")
    #     continue
    # while case_converged==False:
    try:
        print('')
        N_bins, R_max = 500, 10
        atom_onlyion, _ = set_hnc_onlyion(ni, Te, Z, A, Zbar, 
                                    pseudopotential=True, oz_type='svt',r_c=r_c, 
                                    add_bridge=True, bridge='yukawa', N_bins=N_bins, R_max=R_max)
        atom_onlyion.c_s_k_matrix[0,0]*=0
        atom_onlyion.HNC_solve(alpha_method='fixed', alpha_Picard = 5e-1, tol=1e-8, alpha_Ng=0.1, 
                       iters_to_wait=1e4, iters_to_use=2, num_iterations=1e3, verbose=True)

        options={'eps':1e-6,'maxfev':50000,'factor':1,'xtol':1e-8} 
        newton_kwargs= {'method':'hybr', 'options':options} 
        sol = atom_onlyion.HNC_newton_solve( **newton_kwargs)
        print("Only Ion Err: ", atom_onlyion.total_err(sol.x.reshape(1,1,N_bins)))

        atom, qsp = set_hnc(ni, Te, Z, A, Zbar, 
                        pseudopotential=True, oz_type='svt',r_c=r_c, 
                        add_bridge=True, bridge='yukawa',N_bins=N_bins, R_max=R_max)

        atom.c_s_k_matrix[0,0] = 0.5*atom_onlyion.c_s_k_matrix[0,0]

        Picard_converged = atom.HNC_solve(alpha_method='fixed', alpha_Picard = 0.1, tol=1e-3, alpha_Ng=1e-10, 
                       iters_to_wait=1e4, num_iterations=1e3)
        
        options={'eps':1e-6,'maxfev':50000,'factor':100,'xtol':1e-5} 
        newton_kwargs= {'method':'hybr', 'options':options} 
        sol = atom.HNC_newton_solve( **newton_kwargs)
        newton_final_err = atom.total_err(sol.x.reshape(2,2,N_bins))
        print("Root Final Err: ", newton_final_err)
        
        newton_converged = (newton_final_err<1e-3) #atom.newton_succeed
        case_converged = newton_converged

        atom.invert_HNC_OZ([1])
        atom.βueff_r_matrix[0,0] = atom.βueff_r_matrix[0,0] + atom.Bridge_function_Yukawa(atom.r_array, qsp.Γii, qsp.get_κ())

    except Exception as err:
        print("Running Case ID: {0} broken: {1}. Skipping.".format(case_id, err) )
        case_converged=False
        # break
    print("Case time: {0:.3e} s\n".format(time()-t0))
    case_successes[case_id] = case_converged
    atom_case_successes[case_id] = case_converged
    
# a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g)

    if case_converged==True:
        g_r_header = "atom {0} T={1:.1f} eV, r_i={2:.2f}, Zbar (TFDW) ={3:.3f} , solve_err={4:.3e}\n r/r_i \t\t\t\t gii(r/r_i) \t\t\t\t gie(r/r_i) \t\t\t\t gee(r/r_i)".format(case_id, Te/eV_to_AU, Zbar, qsp.ri,newton_final_err)
        np.savetxt("./fits/{0}_g_r.txt".format(case_id), np.array([atom.r_array, atom.h_r_matrix[0,0]+1, atom.h_r_matrix[0,1]+1, atom.h_r_matrix[1,1]+1]).T, header=g_r_header)
        βueff_r_header = "atom {0} T={1:.1f} eV, r_i={2:.2f}, Zbar (TFDW) ={3:.3f} , solve_err={4:.3e}\n r/r_i \t\t\t\t βuii_eff(r/r_i)".format(case_id, Te/eV_to_AU, Zbar, qsp.ri, newton_final_err)
        np.savetxt("./fits/{0}_βu_eff_rawdata.txt".format(case_id), np.array([atom.r_array, atom.βueff_r_matrix[0,0]]).T, header=βueff_r_header)
        try:
            fig, ax = plt.subplots(figsize=(8,6),facecolor='w')
            fig.suptitle(r"atom {0} $T=${1:.1f} eV, $r_i$={2:.2f}, solve_err={3:.3e}".format(case_id, Te/eV_to_AU, qsp.ri, newton_final_err), fontsize=20)

            yukawa_matrix = (atom.Gamma[:,:,np.newaxis]/atom.r_array * np.exp(-atom.r_array*qsp.get_κ())[np.newaxis,np.newaxis,:] ) [:-1,:-1]
            coulomb_matrix = (atom.Gamma[:,:,np.newaxis]/atom.r_array) [:-1,:-1]

            # fit = βu_fit(yukawa_plus_gaussian, atom.r_array, atom.βueff_r_matrix[0,0], initial_guess=[   qsp.Γii, qsp.get_κ(),2 , 1,    -1, 1.9, 1,    1, 0.01, 1 , 10, 2])
            ax.plot(atom.r_array, atom.βu_r_matrix[0,0], 'k--',label='Initial')
            ax.plot(atom.r_array, yukawa_matrix[0,0],'k-.', label="Yukawa")

            ax.plot(atom.r_array, atom.βueff_r_matrix[0,0],color=colors[0], label='Effective')
            # ax.plot(atom.r_array, fit.y/fit.y_fit, color=colors[4],linestyle='--', label="u - fit")
            try: 
                fit1 = βu_fit(yukawa_plus, atom.r_array, atom.βueff_r_matrix[0,0], initial_guess=[   qsp.Γii, qsp.get_κ(),2 , 1])
                ax.plot(atom.r_array, fit1.y_fit, color=colors[1],linestyle='-', label="Yukawa Sigmoid Fit")
                ax.plot(atom.r_array, fit1.y_fit-fit1.y, color=colors[1],linestyle='--', label="Δ Fit")
                np.savetxt("./fits/{0}_βueff.txt".format(case_id), fit1.y_vals[0], header="βu_eff = a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))), fit_err={0:.3e}, solve_err={1:.3e}".format(fit1.err, newton_final_err))
                try: 
                    fit2 = βu_fit(yukawa_plus_cos, atom.r_array, atom.βueff_r_matrix[0,0], initial_guess=[  *fit1.y_vals[0], 1, 0.01, 1 , 10, 2])
                    ax.plot(atom.r_array, fit2.y_fit, color=colors[2],linestyle='-', label="Yukawa Sigmoid Cosine Fit")
                    ax.plot(atom.r_array, fit2.y_fit-fit2.y, color=colors[2],linestyle='--')
                    np.savetxt("./fits/{0}_βueff.txt".format(case_id), fit2.y_vals[0], header="βu_eff = a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + h*np.cos((r-i)*j*np.exp(-k*r))*np.exp(-l*r), fit_err={0:.3e}, solve_err={1:.3e}".format(fit2.err, newton_final_err))
                except:
                    pass
                try:
                    fit3 = βu_fit(yukawa_plus_gaussian_cos, atom.r_array, atom.βueff_r_matrix[0,0], initial_guess=[  *fit2.y_vals[0], -1, 1.9, 1])
                    ax.plot(atom.r_array, fit3.y_fit, color=colors[3],linestyle='-', label="Yukawa Sigmoid Cosine Gaussian Fit")    
                    ax.plot(atom.r_array, fit3.y_fit-fit3.y, color=colors[3],linestyle='--')    
                    np.savetxt("./fits/{0}_βueff.txt".format(case_id), fit3.y_vals[0], header="βu_eff = a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.cos((r-f)*g*np.exp(-h*r))*np.exp(-i*r) + j*np.exp(-(k-r)**2/l), fit_err={0:.3e}, solve_err={1:.3e}".format(fit3.err, newton_final_err))
                except:
                    pass
                # ax.set_title("a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g) \n {0}".format(fit.y_vals[0]))
                # ax.set_title("a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g) + h*np.cos(r*i*np.exp(-j*r))*np.exp(-k*r) \n {0}".format(fit.y_vals[0]))
                # ax.plot(atom.r_array, fit1.y-fit1.y_fit, color=colors[4],linestyle='-', label="u - fit")
                # ax.plot(atom.r_array, fit2.y-fit2.y_fit, color=colors[4],linestyle='-', label="u - fit")
            
            except Exception as err:
                print(err, "Continuing without plotting fit")
                ax.set_title("No fit found")
            
            ax.set_ylim(np.min([-10, 2*np.min(atom.βueff_r_matrix[0,0])]), np.max([1e3, 2*np.max(atom.βueff_r_matrix[0,0])]))
            ax.tick_params(labelsize=20)
            ax.set_xlabel(r"$r/r_s$",fontsize=20)
            ax.set_ylabel(r'$\beta u(r/r_s)$',fontsize=20)
            ax.set_xlabel(r'$r/r_s$',fontsize=20)
           
            ax.set_xlim(0, atom.R_max)
            ax.set_yscale('symlog',linthresh=0.1)
            ax.tick_params(labelsize=15)
            ax.legend(fontsize=10)
            fig.tight_layout()
            fig.savefig("./media/{0}_βueff.png".format(case_id), dpi=400)
            

            fig, axs = plt.subplots(2,2, figsize=(16,10))

            for i in range(2):
                for j in range(2):
                    axs[i,j].plot(atom.r_array, atom.h_r_matrix[i,j]+1)
                    axs[i,j].set_ylabel(r'$g(r)$',fontsize=20)
                    axs[i,j].set_xlabel(r'$r/r_s$',fontsize=20)
                    axs[i,j].tick_params(labelsize=20)
                    axs[i,j].set_xlim(0,atom.R_max)
                    # axs[i,j].legend(fontsize=15)
                    axs[i,j].set_title(atom.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j, atom.Gamma[i][j]) ,fontsize=15)


            fig.tight_layout()
            fig.savefig("./media/{0}_g_matrix.png".format(case_id), dpi=400)
        except Exception as Err:
            print("Not Plotting")
            print("Err")


print("Time to finish = {0:.3e} s".format(time()-t0))
print("α = {0:.3f}".format(α))
print("atom: ", atom_case_successes)

     