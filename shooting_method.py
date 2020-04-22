import numpy as np
import scipy.integrate as spi
import scipy.optimize as opi
import matplotlib 
import csv

import matplotlib.pyplot as plt

def eqns(y, r):
    """ Differential equation for scalar fields 

    Parameters:
        y (list with reals): current status vector ( a(r), alpha(r), phi(r), pi(r) ) 
	r (real) : current position 

    Returns:
        dydr (list with reals): derivative for y in r 

    """
    a, alpha, phi, pi = y

    dadr = a/(2.0)*( -(a**2-1.0)/r + r*(1.0/alpha**2+1.0)*a**2*phi**2+ pi**2 )
    dalphadr = alpha/(2.0) * ( (a**2-1.0)/r + r*(1.0/alpha**2 - 1.0)*a**2*phi**2+pi**2 ) 
    dphidr = pi 
    dpidr = - ( 1.0 + a**2 - r**2*a**2*phi**2)*pi/r - ( 1.0/alpha**2-1.0)*phi*a**2

    dydr = [dadr,dalphadr,dphidr,dpidr]
    return dydr

# solve

def shoot(alpha0_guess,phi0,r):
    """ Solved differential equation for shooting process.

    Parameters:
        alpha0_guess (real): The lapse value guess at r = rmin 
	phi0 (real) : The phi value at r = rmin

    Returns:
        phi_end (real):. The phi value at r = rmax    

    """
    
    # Define initial data vector 
    y0 = [1, alpha0_guess,phi0,0]
    # Solve differential equaion 
    sol = spi.odeint(eqns, y0, r)
    phi_end = sol[-1,2]	
    
    return phi_end

def radial_walker(alpha0_guess,phi0,rstart,rend,deltaR,N): 
    """ Performs shooting for multiple radii rmax shooting process.

    Parameters:
        alpha0_guess (real) : alpha guess for rmin calculation 
        phi0 (real) : phi value at r = 0 
        rstart (real) : first rmax for which shooting is performed
	rend (real) : maximum rmax for which shooting is performed
	deltaR (real) : stelpsize
	N (real) : number of gridpoints 

    Returns:
        alpha0 (real):. alpha0 at r = rmin 
    """

    eps = 1e-10 # distance from zero
    range_list = np.arange(rstart,rend,deltaR)
    alpha0 = alpha0_guess
    #print(range_list)
    #print(alpha0_guess)
    for R in range_list:
        r = np.linspace(eps, R, N)
        #print(r)
        fun = lambda x: shoot(x,phi0,r)
        root = opi.root(fun,alpha0)
        alpha0 = root.x 

        print("step ",R)
        print("alpha0 ",alpha0)
    
    return alpha0[0]

####################################################################################################

# Resolution of diff eqn 
Rstart = 4
Rend = 20
deltaR = 1
N = 100000
alpha0_guess=0.72
phi0_start=0.11
phi0_end=0.14
dphi0=0.01

r = np.linspace(1e-10, Rend, N)

for phi0 in np.arange(phi0_start,phi0_end,dphi0):
	print("Shoot starting from central phi0 value:",phi0)
	alpha0 = radial_walker(alpha0_guess,phi0,Rstart,Rend,deltaR,N)
	y0 = [1, alpha0 ,phi0,0]
	sol = spi.odeint(eqns, y0, r)
    
	a = sol[:, 0]
	alpha = sol[:, 1]
	phi = sol[:, 2]
	M = r / 2.0*(a**2 - 1.0) / a**2

	plt.plot(r, a, color='b', label='a(r)')
	plt.plot(r, M, color='y', label='M(r)')
	plt.plot(r, alpha, color='g', label='alpha(r)')
	plt.plot(r, phi, color='r', label='phi(r)')
	plt.legend(loc='best')
	plt.xlabel('r')
	plt.grid()

plt.savefig("solution_phi10_" + str(phi0_start) + "-" + str(phi0_end) + ".png")
