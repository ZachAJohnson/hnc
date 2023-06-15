# Python file for running hnc TCCW 2 cases
# Zach Johnson
# June 2023

from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


# from hnc_custom_screeningOZ_multiscale import  HNC_solver
from hnc import  HNC_solver
from qsps import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
  

def run_hnc(n_in_per_cc, T, Z, A, Zstar, num_iterations=1e3, alpha = 1e-3, tol=1e-8, c_k_guess=None , which_Tij='thermal',
            oz_type='standard', method='best', add_bridge=False, bridge='yukawa' ):
    n_in_AU = n_in_per_cc*1e6 *aB**3
    ri = QSP_HNC.rs_from_n(n_in_AU)
    qsp = QSP_HNC(Z, A, Zstar, Te, Ti, ri, Zstar*n_in_AU, which_Tij=which_Tij)

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

    βu_r_matrix = np.array([[qsp.βvii(hnc1.r_array), qsp.βvei(hnc1.r_array)],
                            [qsp.βvei(hnc1.r_array), qsp.βvee(hnc1.r_array)]])

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
    converged = hnc1.HNC_solve(alpha_method=method, alpha_Picard = alpha, alpha_oz = 0e-4, h_max=1e8)

    return hnc1, qsp, converged

mixture_file = "/home/zach/plasma/hnc/data/TCCW_single_species_data.csv"
tccw_mixture_data = read_csv(mixture_file)
tccw_cases = [tccw_mixture_data.iloc[n] for n in range(len(tccw_mixture_data))]

case_successes = {}
SVT_case_successes = {}
R_max = 5
N_bins = 500
α = 0.1
for tccw_case in tccw_cases:
	case_id = tccw_case['Case ID']
	print(case_id)
	ni = tccw_case['Number Density [N/cc]']
	Te = tccw_case['Temperature [eV]']*eV
	Ti = Te
	Z = tccw_case['Atomic Number']
	Zstar = tccw_case['Zbar (TF)']
	A = tccw_case['Atomic Weight [a.u.]']
	print('Case ID: ', case_id)

	CH1SVT, CH1qs , converged  = run_hnc(ni, Te, Z, A, Zstar ,method='fixed',alpha=α, tol=1e-6, num_iterations=1e4, oz_type='svt')
	SVT, CH1qsp, converged  = run_hnc(ni, Te, Z, A, Zstar ,method='fixed',alpha=α, tol=1e-6, num_iterations=1e4)
	case_successes[case_id] = converged
	SVT_case_successes[case_id] = converged

print("α = {0:.3f}".format(α))
print("OZE: ", case_successes)
print("SVT: ", SVT_case_successes)

#
# α = 0.100
# OZE:  {'H1': 1, 'C1': 1, 'Al1': 2, 'Cu1': 1, 'Be1': 1, 'Au1': 2, 'H2': 1, 'H3': 1, 'C2': 3, 'C3': 1, 'Al2': 3, 'Al3': 1, 'Cu2': 3, 'Cu3': 2, 'H11': 3, 'H21': 3, 'H31': 3, 'C11': 3, 'C21': 3, 'C31': 3, 'Al11': 3, 'Al21': 3, 'Al31': 3, 'Cu11': 2, 'Cu21': 2, 'Cu31': 2, 'H12': 1, 'H22': 1, 'H32': 2, 'C12': 1, 'C22': 3, 'C32': 3, 'Al12': 2, 'Al22': 3, 'Al32': 3, 'Cu12': 2, 'Cu22': 2, 'Cu32': 2, 'H13': 1, 'H23': 1, 'H33': 1, 'C13': 1, 'C23': 1, 'C33': 2, 'Al13': 2, 'Al23': 2, 'Al33': 3, 'Cu13': 2, 'Cu23': 2, 'Cu33': 2, 'H14': 1, 'H24': 1, 'H34': 1, 'C14': 1, 'C24': 1, 'C34': 1, 'Al14': 2, 'Al24': 2, 'Al34': 1, 'Cu14': 2, 'Cu24': 2, 'Cu34': 2, 'H15': 1, 'H25': 1, 'H35': 1, 'C15': 1, 'C25': 1, 'C35': 1, 'Al15': 2, 'Al25': 2, 'Al35': 1, 'Cu15': 3, 'Cu25': 3, 'Cu35': 3, 'H16': 1}
# SVT:  {'H1': 1, 'C1': 1, 'Al1': 2, 'Cu1': 1, 'Be1': 1, 'Au1': 2, 'H2': 1, 'H3': 1, 'C2': 3, 'C3': 1, 'Al2': 3, 'Al3': 1, 'Cu2': 3, 'Cu3': 2, 'H11': 3, 'H21': 3, 'H31': 3, 'C11': 3, 'C21': 3, 'C31': 3, 'Al11': 3, 'Al21': 3, 'Al31': 3, 'Cu11': 2, 'Cu21': 2, 'Cu31': 2, 'H12': 1, 'H22': 1, 'H32': 2, 'C12': 1, 'C22': 3, 'C32': 3, 'Al12': 2, 'Al22': 3, 'Al32': 3, 'Cu12': 2, 'Cu22': 2, 'Cu32': 2, 'H13': 1, 'H23': 1, 'H33': 1, 'C13': 1, 'C23': 1, 'C33': 2, 'Al13': 2, 'Al23': 2, 'Al33': 3, 'Cu13': 2, 'Cu23': 2, 'Cu33': 2, 'H14': 1, 'H24': 1, 'H34': 1, 'C14': 1, 'C24': 1, 'C34': 1, 'Al14': 2, 'Al24': 2, 'Al34': 1, 'Cu14': 2, 'Cu24': 2, 'Cu34': 2, 'H15': 1, 'H25': 1, 'H35': 1, 'C15': 1, 'C25': 1, 'C35': 1, 'Al15': 2, 'Al25': 2, 'Al35': 1, 'Cu15': 3, 'Cu25': 3, 'Cu35': 3, 'H16': 1}
