
! This solves the two-component HNC equations
! Michael S. Murillo, last modified 12Dec03

! Length units are ion-sphere radii
! Inputs are gamma=e^2/(a_iT), Z, r_s

program TC_HNC

  implicit none


  integer, parameter :: num_bins=1000, num_iter = 300
  real, parameter :: r_max = 10.0, Four_Pi = 12.5663706144
  real, parameter :: Pi = 3.141592653589793, Th_O_F_Pi=0.238732415
  real, parameter :: Two_Pi_Sq = 19.7392088022, ln_2 = 0.693147182
  real, dimension(num_bins) :: u_s_ii_r, u_s_ee_r, u_s_ei_r, u_s_ie_r
  real, dimension(num_bins) :: u_l_ii_k, u_l_ee_k, u_l_ei_k, u_l_ie_k
  real, dimension(num_bins) :: c_ii_k, c_ee_k, c_ei_k, c_ie_k
  real, dimension(num_bins) :: c_s_ii_k, c_s_ee_k, c_s_ei_k, c_s_ie_k
  real, dimension(num_bins) :: c_s_ii_r, c_s_ee_r, c_s_ei_r, c_s_ie_r
  real, dimension(num_bins) :: N_s_ii_k, N_s_ee_k, N_s_ei_k, N_s_ie_k
  real, dimension(num_bins) :: N_s_ii_r, N_s_ee_r, N_s_ei_r, N_s_ie_r
  real, dimension(num_bins) :: g_ii, g_ee, g_ei, g_ie, b_r
  real :: z_ion, lambda_ii, lambda_ee, lambda_ei, gamma, n_e, n_i
  real :: gamma_ii, gamma_ee, gamma_ei
  real :: r, del_r, k, del_k, alpha, r_s, Pauli, determ, P_temp, U_temp
  integer :: bin, it_loop, bin_r, bin_k
  real :: z_ion_th, m_p
  ! Yukawa bridge function
  real :: b_0, b_1, c_1, c_2, c_3, kappa

  open(unit=10, file='Nii.out', status='unknown')
  open(unit=11, file='Nee.out', status='unknown')
  open(unit=12, file='Nei.out', status='unknown')
  open(unit=13, file='Nie.out', status='unknown')
  open(unit=14, file='gii.out', status='unknown')
  open(unit=15, file='gee.out', status='unknown')
  open(unit=16, file='gei.out', status='unknown')
  open(unit=17, file='gie.out', status='unknown')
  open(unit=18, file='gall.out', status='unknown')

  gamma = 0.30233328737867365   ! = e**2/(a_iT)
  z_ion  = 3.0
  gamma_ii = z_ion**2*gamma
  gamma_ee = gamma
  gamma_ei = z_ion*gamma
  r_s = 3 !0.234 ! a_e/a_B
  kappa = 0*sqrt(3*gamma/250.)
  z_ion_th = z_ion**(1./3.)
  n_e = z_ion*Th_O_F_Pi
  n_i = Th_O_F_Pi
  m_p = 1.0  ! ion mass in proton masses
  Pauli = 1.0  ! 0=no or 1=yes
  lambda_ee = sqrt(gamma/(Pi*r_s*z_ion_th))
  lambda_ii = lambda_ee*sqrt(1./(1836.*m_p))
  lambda_ei = sqrt(gamma/(2.0*Pi*r_s*z_ion_th))
  alpha = 1.0

  ! bridge function parameters (Iyetomi, et al., PRA 46, 1051 (1992))
  ! Do not use for gamma<5.0
  b_0 = 0.258 - 0.0612*log(gamma) + 0.0123*(log(gamma))**2 - 1./gamma
  b_1 = 0.0269 + 0.0318*log(gamma) + 0.00814*(log(gamma))**2
  c_1 = 0.498 - 0.28*log(gamma) + 0.0294*(log(gamma))**2
  c_2 = -0.412 + 0.219*log(gamma) - 0.0251*(log(gamma))**2
  c_3 = 0.0988 - 0.0534*log(gamma) + 0.00682*(log(gamma))**2

  write(*,*) 'Z =', z_ion
  write(*,*) 'r_s =', r_s
  write(*,*) 'gamma =', gamma
  write(*,*) 'Gamma_ii =', gamma_ii
  write(*,*) 'Gamma_ei =', gamma_ei
  write(*,*) 'lambda_ii =', lambda_ii
  write(*,*) 'lambda_ee =', lambda_ee
  write(*,*) 'lambda_ei =', lambda_ei
  write(*,*) 'Pauli =', Pauli
  write(*,*) 'kappa =', kappa

  del_r = r_max/num_bins
  del_k = Pi/r_max

  ! Initialize arrays
  do bin = 1, num_bins

     r = bin*del_r
     k = bin*del_k

     ! Yukawa bridge function
     b_r(bin) = 0.00*gamma*(-b_0 + c_1*r**4 + c_2*r**6 + &
          & c_3*r**8)*exp(-b_1*r**2/b_0)*exp(-0.34289*kappa**1.76)

     u_s_ii_r(bin) = gamma_ii*(exp(-alpha*r)-exp(-r/lambda_ii))/r
     u_s_ee_r(bin) = gamma_ee*(exp(-alpha*r)-exp(-r/lambda_ee))/r &
          & + Pauli*ln_2*exp(-r**2/(Pi*ln_2*lambda_ee**2))
     u_s_ei_r(bin) = -gamma_ei*(exp(-alpha*r)-exp(-r/lambda_ei))/r
     u_s_ie_r(bin) = -gamma_ei*(exp(-alpha*r)-exp(-r/lambda_ei))/r

     ! Pseudopotential
     !     u_s_ei_r(bin) = -z_ion*gamma*(exp(-alpha*r)-exp(-r/lambda_ei))/r*(1-exp(-r**2/0.0713))
     !     u_s_ie_r(bin) = -z_ion*gamma*(exp(-alpha*r)-exp(-r/lambda_ei))/r*(1-exp(-r**2/0.0713))

     u_l_ii_k(bin) = Four_Pi*gamma_ii*alpha**2/(k**2*(k**2+alpha**2))
     u_l_ee_k(bin) = Four_Pi*gamma_ee*alpha**2/(k**2*(k**2+alpha**2))
     u_l_ei_k(bin) = -Four_Pi*gamma_ei*alpha**2/(k**2*(k**2+alpha**2))
     u_l_ie_k(bin) = -Four_Pi*gamma_ei*alpha**2/(k**2*(k**2+alpha**2))

     c_ii_k(bin) = -Four_Pi*gamma_ii*lambda_ii**(-2.0)/ &
          & (k**2*(k**2+lambda_ii**(-2.0)))
     c_ee_k(bin) = -Four_Pi*gamma_ee*lambda_ee**(-2.0)/ &
          & (k**2*(k**2+lambda_ee**(-2.0)))
     c_ei_k(bin) = Four_Pi*gamma_ei*lambda_ei**(-2.0)/ &
          & (k**2*(k**2+lambda_ei**(-2.0)))
     c_ie_k(bin) = Four_Pi*gamma_ei*lambda_ei**(-2.0)/ &
          & (k**2*(k**2+lambda_ei**(-2.0)))

     c_s_ii_k(bin) = c_ii_k(bin) + u_l_ii_k(bin)
     c_s_ee_k(bin) = c_ee_k(bin) + u_l_ee_k(bin)
     c_s_ei_k(bin) = c_ei_k(bin) + u_l_ei_k(bin)
     c_s_ie_k(bin) = c_ie_k(bin) + u_l_ie_k(bin)

     N_s_ii_k(bin) = 0.0
     N_s_ee_k(bin) = 0.0
     N_s_ei_k(bin) = 0.0
     N_s_ie_k(bin) = 0.0

     N_s_ii_r(bin) = 0.0
     N_s_ee_r(bin) = 0.0
     N_s_ei_r(bin) = 0.0
     N_s_ie_r(bin) = 0.0

  end do

  ! Main Iteration Loop

  do it_loop = 1, num_iter

     ! Find N_s(k)
     do bin = 1, num_bins

        determ = (1.0 - n_i*(c_s_ii_k(bin)-u_l_ii_k(bin)))* &
             & (1.0 - n_e*(c_s_ee_k(bin)-u_l_ee_k(bin))) - n_e*n_i* &
             & (c_s_ie_k(bin)-u_l_ie_k(bin))*(c_s_ei_k(bin)-u_l_ei_k(bin))

        N_s_ii_k(bin) = ( (c_s_ii_k(bin)-u_l_ii_k(bin))*(1.0 - &
             & n_e*(c_s_ee_k(bin)-u_l_ee_k(bin))) + n_e*&
             & (c_s_ie_k(bin)-u_l_ie_k(bin))*(c_s_ei_k(bin)-u_l_ei_k(bin)) )/&
             & determ - c_s_ii_k(bin)

        N_s_ee_k(bin) = ( (c_s_ee_k(bin)-u_l_ee_k(bin))*(1.0 - &
             & n_i*(c_s_ii_k(bin)-u_l_ii_k(bin))) + n_i*&
             & (c_s_ie_k(bin)-u_l_ie_k(bin))*(c_s_ei_k(bin)-u_l_ei_k(bin)) )/&
             & determ - c_s_ee_k(bin)

        N_s_ie_k(bin) = ( (c_s_ie_k(bin)-u_l_ie_k(bin))*(1.0 - &
             & n_e*(c_s_ee_k(bin)-u_l_ee_k(bin))) + n_e* &
             & (c_s_ie_k(bin)-u_l_ie_k(bin))*(c_s_ee_k(bin)-u_l_ee_k(bin)) )/&
             & determ - c_s_ie_k(bin)

        N_s_ei_k(bin) = ( (c_s_ei_k(bin)-u_l_ei_k(bin))*(1.0 - &
             & n_i*(c_s_ii_k(bin)-u_l_ii_k(bin))) + n_i* &
             & (c_s_ei_k(bin)-u_l_ei_k(bin))*(c_s_ii_k(bin)-u_l_ii_k(bin)) )/&
             & determ - c_s_ei_k(bin)

     end do


     ! Find N_s(r)
     do bin_r = 1, num_bins

        r = bin_r*del_r
        N_s_ii_r(bin_r) = 0.0
        N_s_ee_r(bin_r) = 0.0
        N_s_ei_r(bin_r) = 0.0
        N_s_ie_r(bin_r) = 0.0

        do bin_k = 1, num_bins

           k = bin_k*del_k

           N_s_ii_r(bin_r) = N_s_ii_r(bin_r) + k*sin(k*r)*N_s_ii_k(bin_k)
           N_s_ee_r(bin_r) = N_s_ee_r(bin_r) + k*sin(k*r)*N_s_ee_k(bin_k)
           N_s_ei_r(bin_r) = N_s_ei_r(bin_r) + k*sin(k*r)*N_s_ei_k(bin_k)
           N_s_ie_r(bin_r) = N_s_ie_r(bin_r) + k*sin(k*r)*N_s_ie_k(bin_k)

        end do

        N_s_ii_r(bin_r) = N_s_ii_r(bin_r)*del_k/(Two_Pi_Sq*r) 
        N_s_ee_r(bin_r) = N_s_ee_r(bin_r)*del_k/(Two_Pi_Sq*r) 
        N_s_ei_r(bin_r) = N_s_ei_r(bin_r)*del_k/(Two_Pi_Sq*r) 
        N_s_ie_r(bin_r) = N_s_ie_r(bin_r)*del_k/(Two_Pi_Sq*r) 

     end do

     ! Find g(r) and C_s(r)
     do bin = 1, num_bins

        g_ii(bin) = exp(-u_s_ii_r(bin) + N_s_ii_r(bin) + b_r(bin))
        g_ee(bin) = exp(-u_s_ee_r(bin) + N_s_ee_r(bin))
        g_ei(bin) = exp(-u_s_ei_r(bin) + N_s_ei_r(bin))
        g_ie(bin) = exp(-u_s_ie_r(bin) + N_s_ie_r(bin))

        c_s_ii_r(bin) = g_ii(bin) - 1.0 - N_s_ii_r(bin)
        c_s_ee_r(bin) = g_ee(bin) - 1.0 - N_s_ee_r(bin)
        c_s_ei_r(bin) = g_ei(bin) - 1.0 - N_s_ei_r(bin)
        c_s_ie_r(bin) = g_ie(bin) - 1.0 - N_s_ie_r(bin)

     end do

     ! Find c_s(k)

     do bin_k = 1, num_bins

        k = bin_k*del_k
        c_s_ii_k(bin_k) = 0.0
        c_s_ee_k(bin_k) = 0.0
        c_s_ei_k(bin_k) = 0.0
        c_s_ie_k(bin_k) = 0.0

        do bin_r = 1, num_bins

           r = bin_r*del_r

           c_s_ii_k(bin_k) = c_s_ii_k(bin_k) + r*sin(k*r)*c_s_ii_r(bin_r)
           c_s_ee_k(bin_k) = c_s_ee_k(bin_k) + r*sin(k*r)*c_s_ee_r(bin_r)
           c_s_ei_k(bin_k) = c_s_ei_k(bin_k) + r*sin(k*r)*c_s_ei_r(bin_r)
           c_s_ie_k(bin_k) = c_s_ie_k(bin_k) + r*sin(k*r)*c_s_ie_r(bin_r)

        end do

        c_s_ii_k(bin_k) = c_s_ii_k(bin_k)*Four_Pi*del_r/k
        c_s_ee_k(bin_k) = c_s_ee_k(bin_k)*Four_Pi*del_r/k
        c_s_ei_k(bin_k) = c_s_ei_k(bin_k)*Four_Pi*del_r/k
        c_s_ie_k(bin_k) = c_s_ie_k(bin_k)*Four_Pi*del_r/k

     end do

  end do ! iteration loop

  ! Write final results and compute pressure
  P_temp = 0.0
  U_temp = 0.0
  do bin = 1, num_bins

     k = bin*del_k
     r = bin*del_r

     write(10,*) k, N_s_ii_k(bin)
     write(11,*) k, N_s_ee_k(bin)
     write(12,*) k, N_s_ei_k(bin)
     write(13,*) k, N_s_ie_k(bin)


     write(14,*) r, g_ii(bin)
     write(15,*) r, g_ee(bin)
     write(16,*) r, g_ei(bin)
     write(17,*) r, g_ie(bin)
     write(18,*) r, g_ee(bin), g_ei(bin), g_ii(bin)

     ! Check for z_bar contributions!!!!!
     P_temp = P_temp + r*( g_ii(bin)*(exp(-r/lambda_ii)*(1+r/lambda_ii)-1) &
          & + g_ee(bin)*(exp(-r/lambda_ee)*(1+r/lambda_ee)-1 &
          & - 2*Pauli/(Pi*lambda_ee**2*Gamma)*r**3* &
          & exp(-r**2/(Pi*lambda_ee**2*ln_2))) &
          &  - 2*g_ei(bin)*(exp(-r/lambda_ei)*(1+r/lambda_ei)-1) )

     U_temp = U_temp + r**2*( & 
          & g_ii(bin)*((1-exp(-r/lambda_ii))/r - &
          & exp(-r/lambda_ii)/(2*lambda_ii))+ &
          & g_ee(bin)*( (1-exp(-r/lambda_ee))/r - &
          & exp(-r/lambda_ee)/(2*lambda_ee) &
          & +Pauli/Gamma*ln_2*exp(-r**2/(Pi*lambda_ee**2*ln_2))*&
          & (r**2/(Pi*ln_2*lambda_ee**2)) ) - &
          & 2*g_ei(bin)*((1-exp(-r/lambda_ei))/r - &
          & exp(-r/lambda_ei)/(2*lambda_ei)) &
          & )

  end do

  write(*,*) 'pressure =', 1 - P_temp*0.25*Gamma*del_r
  write(*,*) 'excess energy =', 0.75*U_temp*del_r*Gamma

end program TC_HNC
