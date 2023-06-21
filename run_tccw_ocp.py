# Python file for running hnc TCCW 2 cases
# Zach Johnson
# June 2023

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from time import time

# from hnc_custom_screeningOZ_multiscale import  HNC_solver
# from hnc import  HNC_solver
from hnc_newton import  HNC_solver
from qsps import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
  
def run_hnc_newton(n_in_per_cc, T, Z, A, Zstar, num_iterations=1e3, alpha = 1e-3, tol=1e-8, c_k_guess=None , which_Tij='thermal',
            oz_type='standard', method='kryov', add_bridge=False, bridge='yukawa' , pseudopotential=False, r_c=0.6,
                  rdiff=1e-8):
    n_in_AU = n_in_per_cc*1e6 *aB**3
    ri = QSP_HNC.rs_from_n(n_in_AU)
    qsp = QSP_HNC(Z, A, Zstar, Te, Ti, ri, Zstar*n_in_AU, which_Tij=which_Tij, r_c=r_c)

    N_species = 2
    Gamma = np.array(  [[qsp.Γii,  qsp.Γei],
                        [qsp.Γei,  qsp.Γee]])


    names = ["Ion-1", "Electron", ] 
    kappa = 1
    rhos = np.array([  3/(4*np.pi), Zstar*3/(4*np.pi) ])
    temps = np.array([qsp.Ti, qsp.Te_c])
    masses= np.array([qsp.m_i, m_e])
    hnc1 = HNC_solver(N_species, Gamma, rhos, temps, masses , tol=tol,
                     kappa_multiscale=5, num_iterations=int(num_iterations), 
                     R_max=R_max, N_bins=N_bins, names=names, dst_type=3, oz_method=oz_type)

    if pseudopotential==True:
        βvei = qsp.βvei_atomic(hnc1.r_array)
    else:
        βvei = qsp.βvei(hnc1.r_array)
    βu_r_matrix = np.array([[qsp.βvii(hnc1.r_array), βvei],
                            [βvei, qsp.βvee(hnc1.r_array)]])

    if add_bridge:
        if bridge=='ocp':
            βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_OCP(hnc1.r_array, qsp.Γii)
            # print("Bridge OCP: ", hnc1.Bridge_function_OCP(hnc1.r_array, qsp.Γii))
        elif bridge=='yukawa':
            βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_Yukawa(hnc1.r_array, qsp.Γii, qsp.get_κ())
            # print("Bridge Yukawa: ", hnc1.Bridge_function_Yukawa(hnc1.r_array, qsp.Γii, qsp.get_κ()))
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
    picard_convergence = hnc1.HNC_solve(alpha_method='fixed', alpha_Picard = alpha, alpha_oz = 0e-4)
    hnc1.HNC_newton_solve(method=method, rdiff=rdiff)

    return hnc1, qsp, [picard_convergence, hnc1.newton_succeed]

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

mixture_file = "/home/zach/plasma/hnc/data/TCCW_single_species_data.csv"
tccw_mixture_data = read_csv(mixture_file)
tccw_cases = [tccw_mixture_data.iloc[n] for n in range(len(tccw_mixture_data))]

case_successes = {}
SVT_case_successes = {}
R_max = 10
N_bins = 1000
max_attempts=1
t0 = time()
for tccw_case in tccw_cases[3:]:
    α = 0.1
    case_converged=False
    case_attempts=0
    case_num= tccw_case[' ']
    case_id = tccw_case['Case ID']
    ni = tccw_case['Number Density [N/cc]']
    Te = tccw_case['Temperature [eV]']*eV
    Ti = Te
    Z = tccw_case['Atomic Number']
    # Zstar = tccw_case['Zbar (TF)']
    Zstar = tccw_case['Zbar (TFDW)']
    A = tccw_case['Atomic Weight [a.u.]']
    r_s = tccw_case['Wigner-Seitz Radius [cm]']
    r_c = tccw_case['Average-Bound Radius [cm]']/r_s
    print(r_s, r_c)
    print('\n______________________________\nCase num: {0} Case ID: {1}'.format(case_num, case_id))
    print("Te = {0:.3e} eV, n_i = {1:.3e} 1/cc, r_c/r_s = {2:.3f}".format(Te/eV, ni, r_c))

    while case_converged==False:
        try:
            print('')
            SVT, qsp , converged  = run_hnc_newton(ni, Te, Z, A, Zstar ,method='krylov',alpha=α, tol=1e-4, 
                num_iterations=1e3, oz_type='svt', pseudopotential=True, 
                r_c = r_c, add_bridge=True, bridge='ocp', rdiff=1e-6)
            case_attempts += 1
            if converged[0] != 0 and case_attempts<max_attempts:
                print("Code is: ", converged)
                α*=0.5
                print("Decreaing α to ", α)

            elif converged[0] == 0 or converged[1]==True:
                print("Converged")
                case_converged=True
            
            elif case_attempts>=max_attempts:
                print("Hit max attempts, skipping case.")
                break
        except Exception as err:
            print("Running Case ID: {0} broken: {1}. Skipping.".format(case_id, err) )
            case_converged=False
            break

    case_successes[case_id] = converged
    SVT_case_successes[case_id] = converged
    SVT.invert_HNC([1])
    fig, ax = plt.subplots(figsize=(8,6),facecolor='w')
    fig.suptitle(r"SVT {0} $T=${1:.1f} eV, $r_i$={2:.2f}".format(case_id, Te/eV, qsp.ri), fontsize=20)

    yukawa_matrix = (SVT.Gamma[:,:,np.newaxis]/SVT.r_array * np.exp(-SVT.r_array*qsp.get_κ())[np.newaxis,np.newaxis,:] ) [:-1,:-1]
    coulomb_matrix = (SVT.Gamma[:,:,np.newaxis]/SVT.r_array) [:-1,:-1]

# a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g)

    if case_converged==True:
        try:
            # fit = βu_fit(yukawa_plus_gaussian, SVT.r_array, SVT.βueff_r_matrix[0,0], initial_guess=[   qsp.Γii, qsp.get_κ(),2 , 1,    -1, 1.9, 1,    1, 0.01, 1 , 10, 2])
            ax.plot(SVT.r_array, SVT.βu_r_matrix[0,0], 'k--',label='Initial')
            ax.plot(SVT.r_array, yukawa_matrix[0,0],'k-.', label="Yukawa")

            ax.plot(SVT.r_array, SVT.βueff_r_matrix[0,0],color=colors[0], label='Effective')
            # ax.plot(SVT.r_array, fit.y/fit.y_fit, color=colors[4],linestyle='--', label="u - fit")
            try: 
                fit1 = βu_fit(yukawa_plus, SVT.r_array, SVT.βueff_r_matrix[0,0], initial_guess=[   qsp.Γii, qsp.get_κ(),2 , 1])
                ax.plot(SVT.r_array, fit1.y_fit, color=colors[1],linestyle='-', label="Yukawa Sigmoid Fit")
                ax.plot(SVT.r_array, fit1.y_fit-fit1.y, color=colors[1],linestyle='--', label="Δ Fit")
                np.savetxt("./fits/{0}_βueff.txt".format(case_id), fit1.y_vals[0], header="βu_eff = a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))), err={0:.3e}".format(fit1.err))
                try: 
                    fit2 = βu_fit(yukawa_plus_cos, SVT.r_array, SVT.βueff_r_matrix[0,0], initial_guess=[  *fit1.y_vals[0], 1, 0.01, 1 , 10, 2])
                    ax.plot(SVT.r_array, fit2.y_fit, color=colors[2],linestyle='-', label="Yukawa Sigmoid Cosine Fit")
                    ax.plot(SVT.r_array, fit2.y_fit-fit2.y, color=colors[2],linestyle='--')
                    np.savetxt("./fits/{0}_βueff.txt".format(case_id), fit2.y_vals[0], header="βu_eff = a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + h*np.cos((r-i)*j*np.exp(-k*r))*np.exp(-l*r), err={0:.3e}".format(fit2.err))
                except:
                    pass
                try:
                    fit3 = βu_fit(yukawa_plus_gaussian_cos, SVT.r_array, SVT.βueff_r_matrix[0,0], initial_guess=[  *fit2.y_vals[0], -1, 1.9, 1])
                    ax.plot(SVT.r_array, fit3.y_fit, color=colors[3],linestyle='-', label="Yukawa Sigmoid Cosine Gaussian Fit")    
                    ax.plot(SVT.r_array, fit3.y_fit-fit3.y, color=colors[3],linestyle='--')    
                    np.savetxt("./fits/{0}_βueff.txt".format(case_id), fit3.y_vals[0], header="βu_eff = a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.cos((r-f)*g*np.exp(-h*r))*np.exp(-i*r) + j*np.exp(-(k-r)**2/l), err={0:.3e}".format(fit3.err))
                except:
                    pass
                # ax.set_title("a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g) \n {0}".format(fit.y_vals[0]))
                # ax.set_title("a/r*np.exp(-b*r)/(1+np.exp(c*(r-d))) + e*np.exp(-(f-r)**2/g) + h*np.cos(r*i*np.exp(-j*r))*np.exp(-k*r) \n {0}".format(fit.y_vals[0]))
                # ax.plot(SVT.r_array, fit1.y-fit1.y_fit, color=colors[4],linestyle='-', label="u - fit")
                # ax.plot(SVT.r_array, fit2.y-fit2.y_fit, color=colors[4],linestyle='-', label="u - fit")
            
            except Exception as err:
                print(err, "Continuing without plotting fit")
                ax.set_title("No fit found")
            
            ax.set_ylim(np.min([-10, 2*np.min(SVT.βueff_r_matrix[0,0])]), np.max([1e3, 2*np.max(SVT.βueff_r_matrix[0,0])]))
            ax.tick_params(labelsize=20)
            ax.set_xlabel(r"$r/r_s$",fontsize=20)
            ax.set_ylabel(r'$\beta u(r/r_s)$',fontsize=20)
            ax.set_xlabel(r'$r/r_s$',fontsize=20)
           
            ax.set_xlim(0, R_max)
            ax.set_yscale('symlog',linthresh=0.1)
            ax.tick_params(labelsize=15)
            ax.legend(fontsize=10)
            fig.tight_layout()
            fig.savefig("./media/{0}_βueff.png".format(case_id), dpi=400)
            

            fig, axs = plt.subplots(2,2, figsize=(16,10))

            for i in range(2):
                for j in range(2):
                    axs[i,j].plot(SVT.r_array, SVT.h_r_matrix[i,j]+1)
                    axs[i,j].set_ylabel(r'$g(r)$',fontsize=20)
                    axs[i,j].set_xlabel(r'$r/r_s$',fontsize=20)
                    axs[i,j].tick_params(labelsize=20)
                    axs[i,j].set_xlim(0,R_max)
                    # axs[i,j].legend(fontsize=15)
                    axs[i,j].set_title(SVT.name_matrix[i][j] + r", $\Gamma_{{ {0},{1} }}$ = {2:.2f}".format(i,j, SVT.Gamma[i][j]) ,fontsize=15)


            fig.tight_layout()
            fig.savefig("./media/{0}_g_matrix.png".format(case_id), dpi=400)
        except Exception as Err:
            print("Not Plotting")
            print("Err")


print("Time to finish = {0:.3e} s".format(time()-t0))
print("α = {0:.3f}".format(α))
print("SVT: ", SVT_case_successes)

     