from numpy import pi

π = pi

m_e = 1
m_p = 1833.274 # In AU
aB = 5.29177210903e-11 # Bohr radius in m
k_B = 1.380649e-23 # SI J/K

eV_to_AU = 0.0367512 # So 4 eV = 4 * 0.036.. in natural units
eV_to_K = 11604.5250061598



J_to_erg = 1e7
AU_to_eV = 1/eV_to_AU
AU_to_J = 4.359744e-18
AU_to_erg = AU_to_J*J_to_erg
AU_to_Pa = AU_to_J / aB**3 
AU_to_bar = AU_to_J / aB**3 /1e5 
AU_to_invcc = 1/aB**3/1e6
AU_to_g  = 9.1093837e-28 
AU_to_s  = 2.4188843265e-17
AU_to_kg = 9.1093837e-31
AU_to_K = 1/(8.61732814974493e-5*eV_to_AU) #Similarly, 1 Kelvin = 3.167e-6... in natural units 

cm_to_AU = 1e-2/aB



