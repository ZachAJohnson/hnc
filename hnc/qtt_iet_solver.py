import numpy as np
import tt as ttpy

from liquids.python.qtt_core.qtt_util import * 
from liquids.python.qtt_core.plotting import * 

π = np.pi


class Homogeneous_IET_QTT():
	"""
	This class takes in a number density in grid format
	"""


	def __init__(self, n0, N_bits, eps, r_max, βu_r_func):
		self.N_bits = N_bits
		self.N_grid = 2**N_bits
		self.eps = eps
		self.r_max = r_max
		self.n0 = n0
		self.βu_r_func = βu_r_func

		self.setup_grid()

	def setup_grid(self):
		"""
		Everything that requires referencing a real grid (for now)
		"""
		def qtt_from_tensor(tensor):
		    return ttpy.vector(tensor_to_binarytensor(tensor, order='F'), eps=self.eps )

		self.r_array = np.linspace(0, self.r_max, num=self.N_grid+1)[1:]
		self.dr = self.r_array[1] - self.r_array[0]
		self.k_array = π*(np.arange(self.N_grid)+0.5)/self.r_max
		self.dk = self.k_array[1] - self.k_array[0]
		
		self.r_qtt = qtt_from_tensor(self.r_array)
		self.k_qtt = qtt_from_tensor(self.k_array)

		self.invr_qtt = qtt_from_tensor(1/self.r_array)
		self.invk_qtt = qtt_from_tensor(1/self.k_array)
		
		self.βu_r_qtt = qtt_from_tensor(self.βu_r_func(self.r_array))
		self.f_M_r_qtt = qtt_from_tensor(np.exp(-self.βu_r_func(self.r_array))-1)

		self.one_qtt = ttpy.vector(np.ones([2]*self.N_bits), eps=self.eps)
		self.one_plus_f_M_r_qtt = (self.one_qtt + self.f_M_r_qtt).round(self.eps)

	def round_prod(self,tt1,tt2):
		return (tt1*tt2).round(eps=self.eps)

	def get_γ_r_qtt(self, h_r_qtt, c_r_qtt):
		h_k_qtt = self.FT_r_2_k_qtt(h_r_qtt)
		c_k_qtt = self.FT_r_2_k_qtt(c_r_qtt)
		return self.n0*self.FT_k_2_r_qtt(self.round_prod(h_k_qtt,c_k_qtt))

	def get_h_r_qtt_from_γ_qtt(self, γ_qtt):
		# return (self.f_M_r_qtt + ((self.f_M_r_qtt + self.one_qtt).round(eps=self.eps)*γ_qtt).round(eps=self.eps)).round(eps=self.eps)
		return (self.f_M_r_qtt + (self.one_plus_f_M_r_qtt * γ_qtt).round(eps=self.eps)).round(eps=self.eps)

	def get_c_r_qtt_from_γ_qtt(self, γ_qtt):
		return (self.f_M_r_qtt * (self.one_qtt + γ_qtt).round(eps=self.eps)).round(eps=self.eps)

	def FT_r_2_k_qtt(self, qtt_r):
		return 2*π*self.dr*self.round_prod(self.invk_qtt, dst4_qtt( self.round_prod(self.r_qtt,qtt_r), eps=self.eps))

	def FT_k_2_r_qtt(self, qtt_k):
		return self.N_grid*self.dk/(2*π**2)*self.round_prod(self.invr_qtt, dst4_qtt(self.round_prod(self.k_qtt,qtt_k), eps=self.eps, inverse=True))

	def new_hc(self, h_r_qtt, c_r_qtt ):
		γ_r_qtt = self.get_γ_r_qtt(h_r_qtt, c_r_qtt)
		h_r_qtt = self.get_h_r_qtt_from_γ_qtt(γ_r_qtt)
		c_r_qtt = self.get_c_r_qtt_from_γ_qtt(γ_r_qtt)
		return h_r_qtt, c_r_qtt

	def iterate_h_c(self, h_r_qtt, c_r_qtt, mixing_α, compute_error=True):
		new_h_r_qtt, new_c_r_qtt = self.new_hc(h_r_qtt, c_r_qtt)
		t0 = time()
		res_h = Fr_err( new_h_r_qtt, h_r_qtt )
		res_c = Fr_err( new_c_r_qtt, c_r_qtt )
		print(f"\tError time: {time()-t0:0.3e} s")
		return (self.picard(h_r_qtt, new_h_r_qtt, mixing_α), self.picard(c_r_qtt, new_c_r_qtt, mixing_α)), (res_h, res_c)

	def picard(self, old, new, α):
		return (new*α + (1-α)*old).round(eps=self.eps)

	def solve_OZ(self, h_r_qtt, c_r_qtt, num_iterations=10000, tol=None, mixing_α=0.01, verbose=True):
		if tol is None:
			tol=self.eps
		h_grid = binarytensor_to_tensor(h_r_qtt.full())
		t0 = time()
		t_array = [0]
		float_bits = ttpy.vector.to_list(h_r_qtt)[0].dtype.itemsize
		for self.iter_i in range(num_iterations):
			analysis_time_0=time()
			h_grid = h_r_qtt.full()
			# plot_slice(binarytensor_to_tensor(h_grid+1), self.slice_plot_tuple, save_name=f'../../media/g_iter-{self.iter_i}.png')
			# plot_slice(binarytensor_to_tensor(c_r_qtt.full()), self.slice_plot_tuple, save_name=f'../../media/c_iter-{self.iter_i}.png')
			analysis_time=time()-analysis_time_0
			(h_r_qtt, c_r_qtt), (res_h, res_c) = self.iterate_h_c(h_r_qtt, c_r_qtt, mixing_α)
			self.h_r_qtt, self.c_r_qtt = h_r_qtt, c_r_qtt
			t_array.append(time()-t0)
			if verbose==True:
				print(f"Iter {self.iter_i}, t_tot={t_array[-1]:0.3e} s") 
				print(f"\tδt={t_array[-1]-t_array[-2]:0.3e} s: δt_Analysis={analysis_time:0.3e} s")
				print(f"\tResidual(h)={res_h:0.3e}, Residual(c)={res_c:0.3e}")
				print(f"\tMemory(h)={get_tt_size(h_r_qtt)*float_bits/1e6:0.4f} MB, Memory(c)={get_tt_size(c_r_qtt)*float_bits/1e6:0.4f} MB")
				print(f"\tCompression(h)={get_tt_compression(h_r_qtt):0.3e}, Compression(c)={get_tt_compression(c_r_qtt):0.3e}")
				print(f"\tMin h= {np.min(h_r_qtt.full())}")
				print("\th Ranks:", h_r_qtt.r)
				print("\th Core Sizes:", [np.size(core) for core in ttpy.vector.to_list(h_r_qtt)])
			if res_c<tol and res_h<tol:
				print("Tol reached, BREAK.")
				break 