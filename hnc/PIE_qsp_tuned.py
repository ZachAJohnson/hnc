# Based on the Classical Map Effective Potential model for plasmas
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

from .hnc import  Integral_Equation_Solver as IET_solver
from .qsps import Quantum_Statistical_Potentials as QSPs

from .constants import *
from .misc import rs_from_n, n_from_rs, find_η, ThomasFermiZbar

from scipy.interpolate import LinearNDInterpolator


class Plasma_of_Ions_and_Electrons():
	def __init__(self, Z, A, ni_cc, Ti_in_eV, Te_in_eV, α_params = [1,1,1,1,1], Zbar=None, single_species_guess=False, Picard_max_err = 1,
		βu_options = {}, hnc_options= {}, qsp_options= {}, hnc_solve_options={}, root_options = {}):

		self.hnc_options = { 'N_bins' :500, 'R_max':5, 'oz_method':'svt'}
		self.βu_options  = {'add_bridge':False, 'bridge':'ocp'}
		self.qsp_options = {'r_c':0.6,'which_Tij':'thermal','Te_c_type':'Fermi'}
		self.hnc_solve_options = { 'alpha_Picard': 0.5, 'tol':1e-8, 'alpha_Ng':0, 
                       'iters_to_wait':1e4, 'num_iterations':1e3, 'verbose':False}
		self.root_options = {'method':'hybr', 'options': {'eps':1e-6,'maxfev':10000,'factor':100,'xtol':1e-12}}
		
		self.hnc_options.update(hnc_options)
		self.βu_options.update(βu_options)
		self.qsp_options.update(qsp_options)
		self.hnc_solve_options.update(hnc_solve_options)
		self.root_options.update(root_options)

		# Warnings
		if self.hnc_options['oz_method']=='svt' and self.qsp_options['which_Tij']=='geometric':
			print("WARNING: Inconsistency. Cannot do SVT with geometric interspecies temperature.")

		# Initialize class parameters 
		self.Z, self.A, self.Zbar = Z, A, Zbar
		self.ni_cc = ni_cc
		self.ni_AU = ni_cc/cm_to_AU**3
		self.Te_eV = Te_in_eV
		self.Ti_eV = Ti_in_eV
		self.Te = Te_in_eV*eV_to_AU
		self.Ti = Ti_in_eV*eV_to_AU
		print("Te_in_eV: {0:.3f}".format(Te_in_eV))
		print("Ti_in_eV: {0:.3f}".format(Ti_in_eV))
		self.r_c = self.qsp_options['r_c']
		self.α_Λei, self.α_Λee, self.α_Pauli, self.α_Γei, self.α_Γee = α_params

		if Zbar==None:
			self.Zbar = ThomasFermiZbar(self.Z, self.ni_cc, Te_in_eV)
			print("Te: {0:.3f} eV, Zbar = {1:.3f}".format(Te_in_eV, self.Zbar))

		self.names = ["ion", "electron"] 
		self.Picard_max_err = Picard_max_err
		self.qsp = self.make_qsp(self.ni_cc, self.Zbar, self.Ti, self.Te)
		self.hnc = self.make_hnc(self.qsp, self.Zbar)

	
		self.make_βu_matrix_from_qsp(self.hnc, self.qsp)

	def make_βu_matrix_from_qsp(self, hnc, qsp):
		add_bridge, bridge = self.βu_options['add_bridge'], self.βu_options['bridge']
		r_array = hnc.r_array	
		
		βvei = qsp.βv_Deutsch(self.α_Γei * qsp.Γei, r_array, self.α_Λei*qsp.Λei)  #qsp.βvei(r_array)
		
		βv_Pauli = -np.log(1 - 0.5*self.α_Pauli*np.exp(-r_array**2/(π*self.α_Λee * qsp.Λee**2)) )  
		βvee = qsp.βv_Deutsch(self.α_Γee * qsp.Γee, r_array, self.α_Λee * qsp.Λee) + βv_Pauli 
		
		βvii = qsp.βvii(r_array) + (self.Z-self.Zbar)**2 * βv_Pauli 

		βu_r_matrix = np.array([[βvii, βvei],
		                        [βvei, βvee]])
		if add_bridge:
			if bridge=='ocp':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - IET_solver.Bridge_function_OCP(r_array, qsp.Γii)
			elif bridge=='yukawa':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - IET_solver.Bridge_function_Yukawa(r_array, qsp.Γii, qsp.get_κ())
		
		hnc.set_βu_matrix(βu_r_matrix)
		hnc.initialize_c_k()

	def make_qsp(self, ni_cc, Zbar, Ti, Te):
		n_in_AU = ni_cc*1e6 *aB**3
		ri = rs_from_n(n_in_AU)
		qsp = QSPs(self.Z, self.A, Zbar, Te, Ti, ri, Zbar*n_in_AU, **self.qsp_options)
		return qsp

	def make_hnc(self, qsp, Zbar):
		densities_in_rs = np.array([  3/(4*π), Zbar * 3/(4*π) ])
		temperature_matrix_AU = qsp.Tij
		masses= np.array([qsp.m_i, m_e])
		hnc = IET_solver(2, qsp.Γ_matrix, densities_in_rs, temperature_matrix_AU, masses, **self.hnc_options)
		return hnc


	# Solving HNC system of equations
	def run_ocp_hnc(self):
		self.ocp_hnc = IET_solver(1, self.qsp.Γ_matrix[:1,:1], np.array([3/(4*π)]), np.array([[self.qsp.Ti]]), np.array([self.qsp.m_i]), **self.hnc_options)

		pseudopotential, add_bridge, bridge = self.βu_options[ 'pseudopotential' ], self.βu_options['add_bridge'], self.βu_options['bridge']

		βvii = self.qsp.βvii(self.ocp_hnc.r_array)
		βu_r_matrix = np.array([[βvii]])
		if add_bridge:
			if bridge=='ocp':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - IET_solver.Bridge_function_OCP(self.ocp_hnc.r_array, self.qsp.Γii)
			elif bridge=='yukawa':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - IET_solver.Bridge_function_Yukawa(self.ocp_hnc.r_array, self.qsp.Γii, self.qsp.get_κ())
		
		self.ocp_hnc.set_βu_matrix(βu_r_matrix)
		self.ocp_hnc.initialize_c_k()
		self.ocp_hnc.set_C_matrix()
		self.ocp_hnc.c_s_k_matrix *= 0
		self.ocp_hnc.HNC_solve(**self.hnc_solve_options)

	# Solving HNC system of equations
	def run_yukawa_hnc(self):
		self.yuk_hnc = IET_solver(1, self.qsp.Γ_matrix[:1,:1], np.array([3/(4*π)]), np.array([[self.qsp.Ti]]), np.array([self.qsp.m_i]), kappa = self.qsp.get_κ(), **self.hnc_options)

		pseudopotential, add_bridge, bridge = self.βu_options[ 'pseudopotential' ], self.βu_options['add_bridge'], self.βu_options['bridge']

		βu_r_matrix = self.yuk_hnc.βu_r_matrix.copy()
		if add_bridge:
			if bridge=='ocp':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - IET_solver.Bridge_function_OCP(self.yuk_hnc.r_array, self.qsp.Γii)
			elif bridge=='yukawa':
				βu_r_matrix[0,0] = βu_r_matrix[0,0] - IET_solver.Bridge_function_Yukawa(self.yuk_hnc.r_array, self.qsp.Γii, self.qsp.get_κ())

		self.yuk_hnc.set_βu_matrix(βu_r_matrix)
		self.yuk_hnc.initialize_c_k()
		self.yuk_hnc.set_C_matrix()
		self.yuk_hnc.initialize_βu_matrix()
		self.yuk_hnc.HNC_solve(**self.hnc_solve_options)

	def run_jellium_hnc(self, ideal=True, c_s_k_guess = None):
		self.jellium_hnc = IET_solver(1, self.qsp.Γ_matrix[-1:,-1:], np.array([self.Zbar*3/(4*π)]), np.array([[self.qsp.Te_c]]), np.array([m_e]), **self.hnc_options)
		self.jellium_hnc.c_s_k_matrix *= 0

		# What pauli potential to use
		r_array = self.jellium_hnc.r_array
		if self.find_βuee==True:
			βv_ee_P = self.βP_ee# + self.qsp.βvee(r_array)-self.qsp.βv_Pauli(r_array)
		else:
			βv_ee_P = self.qsp.βv_Pauli(r_array)
			
		# ideal fermi gas or not
		if ideal==True:
			βv_ee = βv_ee_P
		else:
			βv_ee = βv_ee_P + self.qsp.βvee(r_array) - self.qsp.βv_Pauli(r_array)
		
		self.jellium_hnc.set_βu_matrix(np.array([[  βv_ee  ]]))
		
		# Use guess if it exists
		if c_s_k_guess is not None:
			self.jellium_hnc.c_s_k_matrix = c_s_k_guess
		else:
			self.jellium_hnc.c_s_k_matrix *= 0 

		# run
		self.jellium_hnc.HNC_solve(**self.hnc_solve_options)

	def run_hnc(self, hnc=None, qsp=None, c_s_k_guess=None, newton=False):
		if hnc==None and qsp==None:
			hnc = self.hnc
			qsp = self.qsp

		if c_s_k_guess is not None:
			hnc.c_s_k_matrix = c_s_k_guess
		else:
			hnc.c_s_k_matrix *= 0 

		converged = hnc.HNC_solve(**self.hnc_solve_options)

		hnc.invert_HNC_OZ([1])
		
		if newton==False:
			pass
		elif converged!=0 and hnc.final_Picard_err < self.Picard_max_err:
			hnc.HNC_newton_solve( **self.root_options)
		elif hnc.final_Picard_err > self.Picard_max_err:
			print("QUIT: Picard Err too high. Newton assumed not to converge. Try better initial condition or smaller α.")

		# hnc.Heff = self.get_effective_ion_H(hnc, qsp)
		# hnc.Ueff, hnc.Peff = self.get_effective_ion_U_P(hnc, qsp)
		# hnc.u_energy_density = hnc.total_energy_density()/self.qsp.ri**3 
		# hnc.Pressure = hnc.total_pressure()/self.qsp.ri**3 
		hnc.invert_HNC_OZ([1]) # make effective ion plasma
		                        
		if self.βu_options['add_bridge']:
			if self.βu_options['bridge']=='ocp':
				self.βueff_r_matrix_with_B = np.array([[hnc.βueff_r_matrix[0,0] + IET_solver.Bridge_function_OCP(hnc.r_array, qsp.Γii)]])
			elif self.βu_options['bridge']=='yukawa':
				self.βueff_r_matrix_with_B = np.array([[hnc.βueff_r_matrix[0,0] + IET_solver.Bridge_function_Yukawa(hnc.r_array, qsp.Γii, qsp.get_κ())]])

	