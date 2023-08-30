# Based on the Classical Map Effective Potential model for plasmas

from hnc_Ng import  HNC_solver
from qsps import *
from constants import *
from misc import find_η

from atomic_forces.atomOFDFT.python.physics import ThomasFermi
from scipy.optimize import root


from scipy.interpolate import LinearNDInterpolator


class Two_Electron_Plasma():
	def __init__(self, Z, A, ni_cc, Ti_in_eV, Te_in_eV, n_up_fraction = 0.5, Zbar=None, single_species_guess=False, Picard_max_err = 1,
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

		self.Z, self.A, self.Zbar = Z, A, Zbar
		self.n_up_fraction   = n_up_fraction
		self.n_down_fraction = 1-n_up_fraction

		self.ni_cc = ni_cc
		self.Te = Te_in_eV*eV_to_AU
		self.Ti = Ti_in_eV*eV_to_AU
		print("Te_in_eV: {0:.3f}".format(Te_in_eV))
		print("Ti_in_eV: {0:.3f}".format(Ti_in_eV))
		self.r_c = self.qsp_options['r_c']

		if Zbar==None:
			self.Zbar = self.ThomasFermiZbar(self.Z, self.ni_cc, Te_in_eV)
			print("Te: {0:.3f} eV, Zbar = {1:.3f}".format(Te_in_eV, self.Zbar))

		self.names = ["ion", "up electrons", "down electrons"] 
		self.Picard_max_err = Picard_max_err
		self.qsp = self.make_qsp(self.ni_cc, self.Zbar, self.Ti, self.Te)
		self.hnc = self.make_hnc(self.qsp, self.Zbar)
		print("Warning, small T<<EF disagree with DW. Need to adjust?")
		self.get_βPauli()
		self.make_βu_matrix_from_qsp(self.hnc, self.qsp)
		# self.run_hnc(self.hnc, self.qsp)
	
	@staticmethod
	def ThomasFermiZbar( Z, num_density, T):
		"""
		Finite Temperature Thomas Fermi Charge State using 
		R.M. More, "Pressure Ionization, Resonances, and the
		Continuity of Bound and Free States", Adv. in Atomic 
		Mol. Phys., Vol. 21, p. 332 (Table IV).

		Z = atomic number
		num_density = number density (1/cc)
		T = temperature (eV)
		"""

		alpha = 14.3139
		beta = 0.6624
		a1 = 0.003323
		a2 = 0.9718
		a3 = 9.26148e-5
		a4 = 3.10165
		b0 = -1.7630
		b1 = 1.43175
		b2 = 0.31546
		c1 = -0.366667
		c2 = 0.983333

		convert = num_density*1.6726e-24
		R = convert/Z
		T0 = T/Z**(4./3.)
		Tf = T0/(1 + T0)
		A = a1*T0**a2 + a3*T0**a4
		B = -np.exp(b0 + b1*Tf + b2*Tf**7)
		C = c1*Tf + c2
		Q1 = A*R**B
		Q = (R**C + Q1**C)**(1/C)
		x = alpha*Q**beta

		return Z*x/(1 + x + np.sqrt(1 + 2.*x))


	def make_βu_matrix_from_qsp(self, hnc, qsp):
		pseudopotential, add_bridge, bridge = self.βu_options[ 'pseudopotential' ], self.βu_options['add_bridge'], self.βu_options['bridge']
		r_array = hnc.r_array
		if pseudopotential==True:
			βvei = qsp.βvei_atomic(r_array)
		else:
			βvei = qsp.βvei(r_array)
		
		βu_r_matrix = np.array([[qsp.βvii(r_array), qsp.βvei(r_array), qsp.βvei(r_array) ],
		                        [qsp.βvei(r_array), qsp.βv_Deutsch(qsp.Γee, r_array, qsp.Λee) + self.βP_uu, qsp.βvee(r_array)],
		                        [qsp.βvei(r_array), qsp.βvee(r_array), qsp.βv_Deutsch(qsp.Γee, r_array, qsp.Λee) + self.βP_uu]])

		# βu_r_matrix = np.array([[qsp.βvii(r_array), qsp.βvei(r_array), qsp.βvei(r_array) ],
		#                         [qsp.βvei(r_array), qsp.βvee(r_array), qsp.βvee(r_array) ],
		#                         [qsp.βvei(r_array), qsp.βvee(r_array), qsp.βvee(r_array) ]])

		print("Warning, setting qsp based on self βP")
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
		qsp = QSP_HNC(self.Z, self.A, Zbar, Te, Ti, ri, Zbar*n_in_AU, **self.qsp_options)
		return qsp

	def make_hnc(self, qsp, Zbar):
		densities_in_rs = np.array([  3/(4*π), Zbar*self.n_up_fraction* 3/(4*π), Zbar*self.n_down_fraction*3/(4*π) ])
		temperatures_AU = np.array([qsp.Ti, qsp.Te_c, qsp.Te_c])
		masses= np.array([qsp.m_i, m_e, m_e])
		Γ_matrix = np.array([[ qsp.Γii, qsp.Γei , qsp.Γei],
					   [ qsp.Γei, qsp.Γee , qsp.Γee],
					   [ qsp.Γei, qsp.Γee , qsp.Γee]])

		hnc = HNC_solver(3, Γ_matrix, densities_in_rs, temperatures_AU, masses, **self.hnc_options)
		return hnc

	def run_onlyion_hnc(self):
		self.onlyion_hnc = HNC_solver(1, self.qsp.Γ_matrix[:1,:1], np.array([3/(4*π)]), np.array([self.qsp.Ti]), np.array([self.qsp.m_i]), **self.hnc_options)
		self.onlyion_hnc.c_s_k_matrix *= 0
		self.onlyion_hnc.HNC_solve(**self.hnc_solve_options)


	def get_η(self):
		self.η = find_η(self.Te, self.qsp.ne)

	def h_uu_ID_of_r(self, ne, T, r, η):
		sin_arg = np.sqrt(2*T*m_e)*r
		t_max = 5*η

		t = np.linspace(0,t_max, num=10000) 
		dt = t[1]-t[0]
		κ = 3*(2*T*m_e) / (self.qsp.k_F**3 * r)  * np.sum(dt* t*np.sin(sin_arg*t) /(1+np.exp(t**2-η) )  )
		#     h_ee = -0.5*κ**2
		h_uu_ID = -κ**2
		return h_uu_ID

	def get_hee_ID(self):
		self.get_η()
		self.h_uu_ID = np.array([self.h_uu_ID_of_r(self.qsp.ne, self.Te, r , self.η) for r in self.hnc.r_array])
		self.h_ee_ID = self.h_uu_ID/2

	def get_βPauli(self): #Pauli potential
		self.get_hee_ID()
		h_r = self.h_uu_ID.copy() # could do ee instead
		h_k = self.hnc.FT_r_2_k(h_r)

		I_plus_h_rho_inverse = 1/(1 + h_k*self.hnc.rho[0])
		# I_plus_h_rho_inverse = 1/(1 + self.n_up_fraction*h_k*self.hnc.rho[0]) #???
		c_k = I_plus_h_rho_inverse * h_k
		c_r = self.hnc.FT_k_2_r(c_k)

		# Approximate with HNC
		self.βP_uu = h_r - c_r - np.log(h_r+1)
	   
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
		elif hnc.final_Picard_err > self.Picard_max_err:
			print("QUIT: Picard Err too high. Newton assumed not to converge. Try better initial condition or smaller α.")

		hnc.Heff = self.get_effective_ion_H(hnc, qsp)
		hnc.Ueff, hnc.Peff = self.get_effective_ion_U_P(hnc, qsp)
		hnc.u_energy_density = hnc.total_energy_density()/self.qsp.ri**3 
		hnc.Pressure = hnc.total_pressure()/self.qsp.ri**3 


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

	# def get_full_U_P(self, hnc, qsp):
	#     r, dr = qsp.ri*hnc.r_array, qsp.ri*hnc.del_r
	#     Vol_AU = 4/3 * π * (hnc.R_max*qsp.ri)**3

	#     g_r, βveff_r, T, n_AU = hnc.heff_r_matrix[0,0] + 1, hnc.βueff_r_matrix[0,0], qsp.Ti, qsp.ni
	#     veff_r = βveff_r*T
	#     F = -np.gradient(veff_r, r)
	#     veff_r = βveff_r*T

	#     N = Vol_AU*n_AU 
	    
	#     U_ideal = 3/2 * N * T
	#     P_ideal = n_AU * T
	    
	#     U_ex    = 2*π*n_AU*N * np.sum( veff_r * g_r * r**2 * dr, axis=0 )
	#     P_ex    = 2*π/3 * n_AU**2 * np.sum( F * g_r * r**3 * dr, axis=0)
	#     return U_ideal + U_ex, P_ideal + P_ex

	# def get_full_H(self, hnc, qsp):
	# 	Vol_AU = 4/3 * π * (hnc.R_max* qsp.ri)**3
	# 	U, P = self.get_effective_ion_U_P(hnc, qsp)
	# 	return U + P*Vol_AU

	def make_nT_table(self, ε_table=0.01, N_table=2):
		self.mini_atom_table = np.zeros((N_table,N_table)).tolist()
		self.T_table = np.zeros((N_table,N_table))
		self.u_table = np.zeros((N_table,N_table))
		self.Ueff_table = np.zeros((N_table,N_table))
		self.Peff_table = np.zeros((N_table,N_table))
		self.Heff_table = np.zeros((N_table,N_table))

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
				
				self.u_table[i,j] = tmp_hnc.total_energy_density()/self.qsp.ri**3 
				self.Heff_table[i,j]  = tmp_hnc.Heff
				self.Ueff_table[i,j] = tmp_hnc.Ueff
				self.Peff_table[i,j] = tmp_hnc.Peff
				self.T_table[i,j] = Ti

	def get_effective_ion_cp(self, ε_derivative=1e-6, **table_kwargs):
		self.make_nT_table(**table_kwargs)
		
		T_P_data = list(zip( np.array(self.T_table).flatten(),np.array(self.Peff_table).flatten() ))
		H_TP = LinearNDInterpolator(T_P_data, np.array(self.Heff_table).flatten())

		T, P = np.average(self.T_table), np.average(self.Peff_table) 		

		self.Ceff_p_AU = (H_TP(T*(1+ε_derivative), P) - H_TP(T*(1-ε_derivative), P))/(2*T*ε_derivative)
		self.Ceff_v_AU = (self.Ueff_table[0,1]-self.Ueff_table[0,0])/(self.T_table[0,1]-self.T_table[0,0])

		r_s_cc = self.qsp.ri /cm_to_AU
		Vol_cc = 4/3 * π * (self.hnc.R_max*r_s_cc)**3
		N = Vol_cc * self.ni_cc

		Ceff_p_SI = self.Ceff_p_AU * k_B
		Ceff_v_SI = self.Ceff_v_AU * k_B
		
		m_p_SI = 1.672621898e-27 # kg
		M = N  * self.A * m_p_SI 

		self.ceff_p_SI_mass = Ceff_p_SI/M # J/kg/K
		self.ceff_p_SI_vol  = Ceff_p_SI/(Vol_cc*1e-6) #J/m^3/K
		self.ceff_v_SI_vol  = Ceff_v_SI/(Vol_cc*1e-6) #J/m^3/K
		
		n_AU = self.ni_cc * (1e2*aB)**3
		self.Eeff_over_nkBT = self.hnc.Ueff/( self.qsp.Ti*N )
		self.Peff_over_nkBT = self.hnc.Peff/( self.qsp.Ti*n_AU)
		self.ceff_p_over_nkB = self.Ceff_p_AU/N #c_p_SI_vol/k_B/(ni_cc*1e6)
		self.ceff_v_over_nkB = self.Ceff_v_AU/N #c_p_SI_vol/k_B/(ni_cc*1e6)
			

		print("\n_____________________________\nHeat Capacity Results (Effective Ion Picture) ")
		print("c_p = {0:.3e} [J/m^3/K] = {1:.3e} [erg/cc/K]".format(self.ceff_p_SI_vol, self.ceff_p_SI_vol*J_to_erg*1e-6 ))
		print("c_p^ideal = {0:.3e} [J/m^3/K] = {1:.3e} [erg/cc/K]".format(5/2 * self.ni_cc* k_B * 1e6, 5/2 * self.ni_cc * k_B * 1e6*J_to_erg*1e-6 ))
		print("c_v = {0:.3e} [J/m^3/K] = {1:.3e} [erg/cc/K]".format(self.ceff_v_SI_vol, self.ceff_v_SI_vol*J_to_erg*1e-6 ))

		print("\nγ = cp/cv = {0:.3e}".format(self.Ceff_p_AU/self.Ceff_v_AU ))

		print("\nE/nkBT = {0:.3f}, P/nkBT = {1:.3f} ".format(self.Eeff_over_nkBT, self.Peff_over_nkBT))
		print("cp/nkB = {0:.3f}, cv/nkB = {1:.3f} ".format(self.ceff_p_over_nkB, self.ceff_v_over_nkB))

		print("\nTotal cv/nkB estimate (add ideal electrons):")
		print("c_v_tot_estimate = {0:.3f}".format( (self.ceff_v_over_nkB  + self.Zbar*3/2)/(1 + self.Zbar)  ))


	def make_TiTe_table(self, ε_table=0.01, N_table=2, Zbar_fixed=False):
		self.mini_atom_table = np.zeros((N_table,N_table)).tolist()
		self.Te_table = np.zeros((N_table,N_table))
		self.Ti_table = np.zeros((N_table,N_table))
		self.u_table = np.zeros((N_table,N_table))

		Ti_list = self.Ti*np.linspace(1-ε_table, 1+ε_table, num = N_table)
		Te_list = self.Te*np.linspace(1-ε_table, 1+ε_table, num = N_table)
		
		for i, Ti_i in enumerate(Ti_list):
			for j, Te_j in enumerate(Te_list):
				if Zbar_fixed:
					Zbar = self.Zbar
				else:
					Zbar = self.ThomasFermiZbar(self.Z, self.ni_cc, Te_j/eV_to_AU)
					
				print("Te: {0:.3f} eV, Zbar = {1:.3f}".format(Te_j/eV_to_AU, Zbar))

				tmp_qsp = self.make_qsp(self.ni_cc, Zbar, Ti_i, Te_j)
				tmp_hnc = self.make_hnc(tmp_qsp, Zbar)
				
				temp_ratio = self.hnc.Temp_matrix/tmp_hnc.Temp_matrix
				c_s_k_guess = temp_ratio[:,:,np.newaxis] * self.hnc.c_s_k_matrix.copy()
				self.run_hnc(tmp_hnc, tmp_qsp, c_s_k_guess = c_s_k_guess)

				self.mini_atom_table[i][j] = tmp_hnc
				
				self.u_table[i,j]  = tmp_hnc.total_energy_density()/self.qsp.ri**3 
				self.Te_table[i,j] = Te_j
				self.Ti_table[i,j] = Ti_i
	
	def get_cv(self, ε_derivative=1e-6, **table_kwargs):
		self.make_TiTe_table(**table_kwargs)

		self.ci_v_AU = 0.5*np.sum(  (self.u_table[1,:]-self.u_table[0,:])/(self.Ti_table[1,:]-self.Ti_table[0,:])   ) # averages two Te's
		self.ce_v_AU = 0.5*np.sum(  (self.u_table[:,1]-self.u_table[:,0])/(self.Te_table[:,1]-self.Te_table[:,0])   ) # averages two Ti's
		self.c_v_AU = self.ci_v_AU + self.ce_v_AU
		
		self.ci_v_over_nkB = self.ci_v_AU/self.hnc.rho[0] * self.qsp.ri**3 #
		self.ce_v_over_nkB = self.ce_v_AU/self.hnc.rho[1] * self.qsp.ri**3 #
		self.c_v_over_nkB = self.c_v_AU /np.sum(self.hnc.rho) * self.qsp.ri**3
		
		self.ci_v_SI_vol  = self.ci_v_AU*k_B/(aB**3) #J/m^3/K
		self.ce_v_SI_vol  = self.ce_v_AU*k_B/(aB**3) #J/m^3/K
		self.c_v_SI_vol   = self.c_v_AU *k_B/(aB**3) #J/m^3/K
		
		# self.E_over_nkBT = self.hnc.U/( self.qsp.Ti*N )
		
		print("\n_____________________________\nHeat Capacity Results ")
		print("c^e_v = {0:.3e} [J/m^3/K], c^i_v  = {1:.3e} [J/m^3/K] ".format(self.ce_v_SI_vol, self.ci_v_SI_vol))
		print("c_v = {0:.3e} [J/m^3/K] ".format(self.c_v_SI_vol))

		print("\nc^e_v/(ne kB)= {0:.3f} , c^i_v/(ni kB)   = {1:.3f} ".format(self.ce_v_over_nkB, self.ci_v_over_nkB))
		print("c_v/(n kB) = {0:.3f} ".format(self.c_v_over_nkB))

		

		

	    


