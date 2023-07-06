# Based on the Classical Map Effective Potential model for plasmas

from hnc_Ng import  HNC_solver
from qsps import *
from constants import *


from scipy.interpolate import LinearNDInterpolator


class CMEP_Atom():
	def __init__(self,Z, A, ni_cc, Ti_in_eV, Te_in_eV, Zbar=None, single_species_guess=False, Picard_max_err = 1,
		βu_options = {}, hnc_options= {}, qsp_options= {}, hnc_solve_options={}, root_options = {}):

		self.hnc_options = { 'N_bins' :500, 'R_max':5, 'oz_method':'svt'}
		self.βu_options  = {'add_bridge':False, 'bridge':'ocp', 'pseudopotential':True}
		self.qsp_options = {'r_c':0.6}
		self.hnc_solve_options = {'alpha_method':'fixed', 'alpha_Picard': 0.5, 'tol':1e-8, 'alpha_Ng':0, 
                       'iters_to_wait':1e4, 'num_iterations':1e3, 'verbose':False}
		self.root_options = {'method':'hybr', 'options': {'eps':1e-6,'maxfev':10000,'factor':100,'xtol':1e-8}}
		
		self.hnc_options.update(hnc_options)
		self.βu_options.update(βu_options)
		self.qsp_options.update(qsp_options)
		self.hnc_solve_options.update(hnc_solve_options)
		self.root_options.update(root_options)

		if Zbar==None:
			self.Zbar = self.ThomasFermiZbar()

		self.Z, self.A, self.Zbar = Z, A, Zbar
		self.ni_cc = ni_cc
		self.Te = Te_in_eV*eV_to_AU
		self.Ti = Ti_in_eV*eV_to_AU
		
		self.r_c = self.qsp_options['r_c']

		self.names = ["ion", "electron"] 
		
		self.qsp = self.make_qsp(self.ni_cc, self.Zbar, self.Ti, self.Te)
		self.hnc = self.make_hnc(self.qsp, self.Zbar)
		# self.run_hnc(self.hnc, self.qsp)
	
	def make_βu_matrix_from_qsp(self, hnc, qsp):
		pseudopotential, add_bridge, bridge = self.βu_options[ 'pseudopotential' ], self.βu_options['add_bridge'], self.βu_options['bridge']
		r_array = hnc.r_array
		if pseudopotential==True:
			βvei = qsp.βvei_atomic(r_array)
		else:
			βvei = qsp.βvei(r_array)
		
		βu_r_matrix = np.array([[qsp.βvii(r_array), βvei],
		                      [βvei, qsp.βvee(r_array)]])
		if add_bridge:
			if bridge=='ocp':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - HNC_solver.Bridge_function_OCP(r_array, qsp.Γii)
			elif bridge=='yukawa':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - HNC_solver.Bridge_function_Yukawa(r_array, qsp.Γii, qsp.get_κ())
		
		hnc.set_βu_matrix(βu_r_matrix)
		hnc.initialize_c_k()

	def make_qsp(self, ni_cc, Zbar, Ti, Te):
		n_in_AU = ni_cc*1e6 *aB**3
		ri = QSP_HNC.rs_from_n(n_in_AU)
		qsp = QSP_HNC(self.Z, self.A, Zbar, Ti, Te, ri, Zbar*n_in_AU, **self.qsp_options)
		return qsp

	def make_hnc(self, qsp, Zbar):
		densities_in_rs = np.array([  3/(4*π), Zbar * 3/(4*π) ])
		temperatures_AU = np.array([qsp.Ti, qsp.Te_c])
		masses= np.array([qsp.m_i, m_e])
		hnc = HNC_solver(2, qsp.Γ_matrix, densities_in_rs, temperatures_AU, masses, **self.hnc_options)
		self.make_βu_matrix_from_qsp(hnc, qsp)
		return hnc

	def run_onlyion_hnc(self):
		self.onlyion_hnc = HNC_solver(1, self.qsp.Γ_matrix[:1,:1], np.array([3/(4*π)]), np.array([self.qsp.Ti]), np.array([self.qsp.m_i]), **self.hnc_options)
		self.onlyion_hnc.c_s_k_matrix *= 0
		self.onlyion_hnc.HNC_solve(**self.hnc_solve_options)


	def run_hnc(self, hnc=None, qsp=None, c_s_k_guess=None):
		if hnc==None and qsp==None:
			hnc = self.hnc
			qsp = self.qsp

		if c_s_k_guess is not None:
			hnc.c_s_k_matrix = c_s_k_guess
		else:
			hnc.c_s_k_matrix *= 0 

		converged = hnc.HNC_solve(**self.hnc_solve_options)

		hnc.invert_HNC_OZ([1])
		
		if converged!=0 and hnc.final_Picard_err < self.Picard_max_err:
			hnc.HNC_newton_solve( **self.root_options)
		elif hnc.final_Picard_err < self.Picard_max_err:
			print("QUIT: Picard Err too low. Newton assumed not to converge.")

		hnc.Heff = self.get_effective_ion_H(hnc, qsp)
		hnc.Ueff, hnc.Peff = self.get_effective_ion_U_P(hnc, qsp)

	def get_effective_ion_U_P(self, hnc, qsp):
	    r, dr = qsp.ri*hnc.r_array, qsp.ri*hnc.del_r
	    Vol_AU = 4/3 * π * (hnc.R_max*qsp.ri)**3

	    g_r, βveff_r, T, n_AU = hnc.heff_r_matrix[0,0] + 1, hnc.βueff_r_matrix[0,0], qsp.Ti, qsp.ni
	    veff_r = βveff_r*T
	    F = -np.gradient(veff_r, r)
	    veff_r = βveff_r*T

	    N = Vol_AU*n_AU 
	    
	    U_ideal = 3/2 * N * T
	    P_ideal = n_AU * T
	    
	    U_ex    = 2*π*n_AU*N * np.sum( veff_r * g_r * r**2 * dr, axis=0 )
	    P_ex    = 2*π/3 * n_AU**2 * np.sum( F * g_r * r**3 * dr, axis=0)
	    return U_ideal + U_ex, P_ideal + P_ex

	def get_effective_ion_H(self, hnc, qsp):
		Vol_AU = 4/3 * π * (hnc.R_max* qsp.ri)**3
		U, P = self.get_effective_ion_U_P(hnc, qsp)
		return U + P*Vol_AU

	def make_mini_table(self, ε_table=0.01, N_table=2):
		self.mini_atom_table = np.zeros((N_table,N_table)).tolist()
		self.H_table = np.zeros((N_table,N_table))
		self.T_table = np.zeros((N_table,N_table))
		self.U_table = np.zeros((N_table,N_table))
		self.P_table = np.zeros((N_table,N_table))

		ni_list = self.ni_cc*np.linspace(1-ε_table, 1+ε_table, num = N_table)
		Ti_list = self.Te*np.linspace(1-ε_table, 1+ε_table, num = N_table)
		Te_list = self.Ti*np.linspace(1-ε_table, 1+ε_table, num = N_table)
		
		for i, ni_i in enumerate(ni_list):
			for j, Ts_j in enumerate(zip(Te_list, Ti_list)):
				Te, Ti = Ts_j
				tmp_qsp = self.make_qsp(ni_i, self.Zbar, Ti, Te)
				tmp_hnc = self.make_hnc(tmp_qsp, self.Zbar)
				
				temp_ratio = self.hnc.Temp_matrix/tmp_hnc.Temp_matrix
				c_s_k_guess = temp_ratio[:,:,np.newaxis] * self.hnc.c_s_k_matrix.copy()
				self.run_hnc(tmp_hnc, tmp_qsp, c_s_k_guess = c_s_k_guess)

				self.mini_atom_table[i][j] = tmp_hnc
				self.H_table[i,j]  = tmp_hnc.Heff
				self.U_table[i][j] = tmp_hnc.Ueff
				self.P_table[i][j] = tmp_hnc.Peff
				self.T_table[i][j] = Ti

	def get_cp(self, ε_derivative=1e-6, **table_kwargs):
		self.make_mini_table(**table_kwargs)
		
		T_P_data = list(zip( np.array(self.T_table).flatten(),np.array(self.P_table).flatten() ))
		H_TP = LinearNDInterpolator(T_P_data, np.array(self.H_table).flatten())

		T, P = np.average(self.T_table), np.average(self.P_table) 		

		self.C_p_AU = (H_TP(T*(1+ε_derivative), P) - H_TP(T*(1-ε_derivative), P))/(2*T*ε_derivative)
		self.C_v_AU = (self.U_table[0,1]-self.U_table[0,0])/(self.T_table[0,1]-self.T_table[0,0])

		r_s_cc = self.qsp.ri /cm_to_AU
		Vol_cc = 4/3 * π * (self.hnc.R_max*r_s_cc)**3
		N = Vol_cc * self.ni_cc

		C_p_SI = self.C_p_AU * k_B
		C_v_SI = self.C_v_AU * k_B
		
		m_p_SI = 1.672621898e-27 # kg
		M = N  * self.A * m_p_SI 

		self.c_p_SI_mass = C_p_SI/M # J/kg/K
		self.c_p_SI_vol  = C_p_SI/(Vol_cc*1e-6) #J/m^3/K
		self.c_v_SI_vol  = C_v_SI/(Vol_cc*1e-6) #J/m^3/K
		
		n_AU = self.ni_cc * (1e2*aB)**3
		self.E_over_nkBT = self.hnc.Ueff/( self.qsp.Ti*N )
		self.P_over_nkBT = self.hnc.Peff/( self.qsp.Ti*n_AU)
		self.c_p_over_nkB = self.C_p_AU/N #c_p_SI_vol/k_B/(ni_cc*1e6)
			

		print("\n_____________________________\nHeat Capacity Results ")
		print("c_p = {0:.3e} [J/m^3/K] = {1:.3e} [erg/cc/K]".format(self.c_p_SI_vol, self.c_p_SI_vol*J_to_erg*1e-6 ))
		print("c_p^ideal = {0:.3e} [J/m^3/K] = {1:.3e} [erg/cc/K]".format(5/2 * self.ni_cc* k_B * 1e6, 5/2 * self.ni_cc * k_B * 1e6*J_to_erg*1e-6 ))
		print("")
		print("c_v = {0:.3e} [J/m^3/K] = {1:.3e} [erg/cc/K]".format(self.c_v_SI_vol, self.c_v_SI_vol*J_to_erg*1e-6 ))
		print("γ = cp/cv = {0:.3e}".format(self.C_p_AU/self.C_v_AU ))
		print("E/nkBT = {0:.3e}, P/nkBT = {1:.3e}, cp/nkB = {2:.3e}".format(self.E_over_nkBT, self.P_over_nkBT, self.c_p_over_nkB))




	    


