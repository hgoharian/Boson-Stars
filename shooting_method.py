import numpy as np
import scipy.integrate as spi
import scipy.optimize as opi
import matplotlib.pyplot as plt



def eqns(y, r):
    """ Differential equation for scalar fields 
    Parameters:
        y (list with reals): current status vector ( a(r), alpha(r), phi(r), pi(r) ) 
		r (real) : current radial position 
    Returns:
        dydr (list with reals): derivative of y with respect to r 
    """
    a, alpha, phi, pi = y

    #Eqns (8.9),(8.10),(8.11):
    dadr = a/(2.0)*( -(a**2-1.0)/r + r*(1.0/alpha**2+1.0)*a**2*phi**2+ pi**2 )
    dalphadr = alpha/(2.0) * ( (a**2-1.0)/r + r*(1.0/alpha**2 - 1.0)*a**2*phi**2+pi**2 ) 
    dphidr = pi 
    dpidr = - ( 1.0 + a**2 - r**2*a**2*phi**2)*pi/r - ( 1.0/alpha**2-1.0)*phi*a**2

    dydr = [dadr,dalphadr,dphidr,dpidr]
    return dydr



def shoot(alphac,phic,r):
	""" differential equations to be solved for shooting process
	
	Parameters:
	    alphac (real): lapse value at r = rmin 
		phic (real): scalar field value at r = rmin
		r (list with reals): list of radial positions
	
	Returns:
	    phi_end (real): scalar field value at r = rmax    
	
	"""
	
	# Boundary conditions at r = rmin i.e. Eq (8.12),(8.13),(8.14) 
	#y(a(r), alpha(r), phi(0), Pi(0)):
	yc = [1, alphac, phic, 0]
	# Solve differential equations 
	sol = spi.odeint(eqns, yc, r)
	phi_end = sol[-1,2]

	return phi_end



def radial_walker(alphac_guess,phic,rstart,rend,deltaR,N): 
	""" Performs shooting for multiple radii rmax shooting process.
	
	Parameters:
	    alphac_guess (real) : initial guess for the lapse value at r = rmin
	    phic (real) : scalar field value at r = 0 
	    rstart (real) : first rmax for which shooting is performed
		rend (real) : maximum rmax for which shooting is performed
		deltaR (real) : stepsize
		N (real) : number of gridpoints 
	
	Returns:
	    alphac (real):. alphac at r = rmin 
	"""
	
	eps = 1e-10 # distance from zero
	range_list = np.arange(rstart,rend,deltaR)
	alphac = alphac_guess
	#print(range_list)
	#print(alphac_guess)
	for R in range_list:
		r = np.linspace(eps, R, N)
		
		# Boundary condition at r = rmax i.e. Eq (8.15)
		fun = lambda x: shoot(x,phic,r) #fun is for function, takes alpha_guess in and spits out the value of phi at rmax, takes in an alpha and spits out phi at r->infty
		root = opi.root(fun,alphac)
		alphac = root.x 
		
		print("step ",R)
		print("alphac ",alphac)
    
	return alphac[0]



####################################################################################################



# runtime parameters
Rstart = 4
Rend = 20
deltaR = 1
N = 100000
alphac_guess=0.72
phic_start=0.1
phic_end=0.50
dphic=0.1

r = np.linspace(1e-10, Rend, N)
#set feq_phic and mass_phic as empty lists
freq_phic = []
mass_phic = []

for phic in np.arange(phic_start,phic_end,dphic):
	print("Shoot starting from central phic value:",phic)
	alphac = radial_walker(alphac_guess,phic,Rstart,Rend,deltaR,N)
	
    # Boundary conditions at r = rmin i.e. Eq (8.12),(8.13),(8.14)
	yc = [1, alphac ,phic,0]
    # Solve differential equations
	sol = spi.odeint(eqns, yc, r)
    
	a = sol[:, 0]
	alpha = sol[:, 1]
	phi = sol[:, 2]
	M = r / 2.0*(a**2 - 1.0) / a**2

	# update lists of frequency and total mass for each phic
	freq_phic.append(1./a[N-1]/alpha[N-1])
	mass_phic.append(M[N-1])

	# output frequency omega according to (8.16) and rescaling \tilde{\alpha}=(m/omega)\alpha
	print("frequency:",1./a[N-1]/alpha[N-1])
	print("mass:",M[N-1])

	# color function
	f0=1+(phic_start-phic)/(phic_end-phic_start)/2.0

	# plot for each phic
	plt.plot(r, a, color=(0,f0,0,1), label='a(r)')
	plt.plot(r, M, color=(0,f0,f0,1), label='M(r)')
	plt.plot(r, alpha, color=(f0,0,0,1),label='alpha(r)')
	plt.plot(r, phi, color=(f0,f0,0,1), label='phi(r)')
	plt.xlabel('r')
	plt.grid()

# plot functions of r 
plt.savefig("solution_phi10_" + str(phic_start) + "-" + str(phic_end) + ".png")

# plot frequency as a function of phic 
plt.clf()
plt.plot(np.arange(phic_start,phic_end,dphic),freq_phic)
plt.xlabel('phic')
plt.ylabel('frequency')
plt.grid()
plt.savefig("freq_phi10_" + str(phic_start) + "-" + str(phic_end) + ".png")

# plot mass as a function of phic 
plt.clf()
plt.plot(np.arange(phic_start,phic_end,dphic),mass_phic)
plt.xlabel('phic')
plt.ylabel('mass')
plt.grid()
plt.savefig("total_mass_phi10_" + str(phic_start) + "-" + str(phic_end) + ".png")
