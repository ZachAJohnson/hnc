from numpy import pi

π = pi

m_e = 1
m_p = 1836.1526724584617 # In AU

α  = 0.0072973525693 # 1/c in AU
c  = 1/α

aB = 5.29177210903e-11 # Bohr radius in m
k_B = 1.380649e-23 # SI J/K

eV_to_AU = 0.03674932539796232 # PDG
eV_to_K = 11604.5250061598
K_to_eV = 1/eV_to_K


J_to_erg = 1e7
AU_to_amu = 1.007276466812/m_p # 1/12 of carbon mass
AU_to_eV = 1/eV_to_AU
AU_to_Ry = 2
AU_to_J = 4.359744e-18
AU_to_erg = AU_to_J*J_to_erg
AU_to_Pa = AU_to_J / aB**3 
AU_to_bar = AU_to_J / aB**3 /1e5 
AU_to_invcc = 1/aB**3/1e6
AU_to_g  = 9.1093837e-28 
AU_to_s  = 2.4188843265e-17
AU_to_kg = 9.1093837e-31
AU_to_K = 1/(8.61732814974493e-5*eV_to_AU) #Similarly, 1 Kelvin = 3.167e-6... in natural units 
AU_to_m = aB
AU_to_cm = aB*1e2
AU_to_Angstrom = AU_to_cm*1e8
AU_to_Coulombs = 1.602176634e-19 #C
AU_to_Amps = AU_to_Coulombs/AU_to_s
AU_to_Volts = 27.211386245988
AU_to_Ohms  = AU_to_Volts/AU_to_Amps
AU_to_Siemens = 1/AU_to_Ohms


amu_to_AU = 1/AU_to_amu
eV_to_AU   = 1/AU_to_eV
Ry_to_AU = 1/AU_to_Ry
J_to_AU   = 1/AU_to_J
erg_to_AU   = 1/AU_to_erg
Pa_to_AU   = 1/AU_to_Pa
bar_to_AU   = 1/AU_to_bar
invcc_to_AU   = 1/AU_to_invcc
g_to_AU   = 1/AU_to_g
s_to_AU   = 1/AU_to_s
kg_to_AU   = 1/AU_to_kg
K_to_AU   = 1/AU_to_K
cm_to_AU = 1/AU_to_cm
m_to_AU = 1/AU_to_m
Angstrom_to_AU = 1/AU_to_Angstrom
Amps_to_AU = 1/AU_to_Amps
Coulombs_to_AU = 1/AU_to_Coulombs
Volts_to_AU = 1/AU_to_Volts
Ohms_to_AU = 1/AU_to_Ohms
Siemens_to_AU = 1/AU_to_Siemens







