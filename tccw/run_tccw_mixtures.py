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
  

def run_mixture_hnc(n_in_per_cc, T, Z1,Z2, A1,A2, Zstar1,Zstar2, num_iterations=1e3, alpha = 1e-3, tol=1e-8, c_k_guess=None , which_Tij='thermal',
            oz_type='standard', method='best', add_bridge=False, bridge='yukawa' ):
    n_in_AU = n_in_per_cc*1e6 *aB**3
    ne = (Zstar1 + Zstar2)*n_in_AU
    ri = QSP_HNC.rs_from_n(n_in_AU)
    qsp_i1 = QSP_HNC(Z1, A1, Zstar1, Te, Ti, ri, ne, which_Tij=which_Tij)
    qsp_i2 = QSP_HNC(Z2, A2, Zstar2, Te, Ti, ri, ne, which_Tij=which_Tij)

    N_species = 3
    Γ0 = 1/Ti/ri
    Gamma = np.array(  [[Zstar1**2*Γ0,      Zstar1*Zstar2*Γ0, qsp_i1.Γei],
                        [Zstar1*Zstar2*Γ0,  Zstar2**2*Γ0,     qsp_i2.Γei],
                        [qsp_i1.Γei,        qsp_i2.Γei,       qsp_i2.Γee]])


    names = ["Ion-1", "Ion-2", "Electron", ] 
    rhos = np.array([  3/(4*np.pi), 3/(4*np.pi), (Zstar1+Zstar2)*3/(4*np.pi) ])
    temps = np.array([qsp_i1.Ti, qsp_i2.Ti, qsp_i1.Te_c])
    masses= np.array([qsp_i1.m_i, qsp_i2.m_i , m_e])
    hnc1 = HNC_solver(N_species, Gamma, rhos, temps, masses , tol=tol,
                     kappa_multiscale=1, num_iterations=int(num_iterations), 
                     R_max=R_max, N_bins=N_bins, names=names, dst_type=3, oz_method=oz_type)
    
    βvii_0 = qsp_i1.βvii(hnc1.r_array)/Zstar1**2
    βu_r_matrix = np.array([[βvii_0*Zstar1**2,         βvii_0*Zstar1*Zstar2,      qsp_i1.βvei(hnc1.r_array)],
                            [βvii_0*Zstar1*Zstar2,     βvii_0*Zstar2**2,          qsp_i2.βvei(hnc1.r_array)],
                            [qsp_i1.βvei(hnc1.r_array),qsp_i2.βvei(hnc1.r_array), qsp_i1.βvee(hnc1.r_array)]])

#     if add_bridge:
#         if bridge=='ocp':
#             βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_OCP(hnc1.r_array, qsp.Γii)
#         elif bridge=='yukawa':
#             βu_r_matrix[0,0] = βu_r_matrix[0,0] - hnc1.Bridge_function_Yukawa(hnc1.r_array, qsp.Γii, qsp.get_κ())
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

    return hnc1, qsp_i1, qsp_i2, converged


mixture_file = "/home/zach/plasma/hnc/data/TCCW_binary_mixture_data.csv"
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
	Z1, Z2 = tccw_case['Atomic Number 1'], tccw_case['Atomic Number 2']
	Zstar1, Zstar2 = tccw_case['Zbar 1 (TF)'], tccw_case['Zbar 2 (TF)'] 
	A1, A2 = tccw_case['Atomic Weight 1 [a.u.]'], tccw_case['Atomic Weight 1 [a.u.]']
	print('Case ID: ', case_id)

	CH1SVT, CH1qsp1, CH1qsp2 , converged  = run_mixture_hnc(ni, Te, Z1, Z2, A1,A2, Zstar1, Zstar2 ,method='fixed',alpha=α, tol=1e-6, num_iterations=1e4, oz_type='svt')
	SVT, CH1qsp1, CH1qsp2 , converged  = run_mixture_hnc(ni, Te, Z1, Z2, A1,A2, Zstar1, Zstar2 ,method='fixed',alpha=α, tol=1e-6, num_iterations=1e4)
	case_successes[case_id] = converged
	SVT_case_successes[case_id] = converged

print("α = {0:.3f}".format(α))
print("OZE: ", case_successes)
print("SVT: ", SVT_case_successes)

# α = 0.100
# OZE:  {'CH1': True, 'HCu1': True, 'CH2': True, 'CH1i1': True, 'CH1i2': True, 'CH1i3': True, 'CH1i4': True, 'CH1i5': True, 'CH1i6': True, 'HCu1i1': True, 'HCu1i2': True, 'HCu1i3': True, 'HCu1i4': False, 'HCu1i5': False, 'HCu1i6': False}
# SVT:  {'CH1': True, 'HCu1': True, 'CH2': True, 'CH1i1': True, 'CH1i2': True, 'CH1i3': True, 'CH1i4': True, 'CH1i5': True, 'CH1i6': True, 'HCu1i1': True, 'HCu1i2': True, 'HCu1i3': True, 'HCu1i4': False, 'HCu1i5': False, 'HCu1i6': False}
