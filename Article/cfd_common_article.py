import numpy as np
import scipy.fftpack as fft
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as lg



def mpl_pause_background (delay):
	"""
	Replaces plt.pause(). The windows stay in background.
	"""
	backend = plt.rcParams['backend']
	if backend in matplotlib.rcsetup.interactive_bk:
		figManager = matplotlib._pylab_helpers.Gcf.get_active()
		if figManager is not None:
			canvas = figManager.canvas
			if canvas.figure.stale:
				canvas.draw()
			canvas.start_event_loop(delay)
########################## Semi-Lagrangian advection routines ##########################

def SemiLag (u,v, q, Δx,Δy, Δt):
	"""
	Semi-lagrangian 2D advection of the scalar field q by (u,v).
	Reslting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""

	# Matrices where 1 is right, 0 is left or center
	Mx2 = np.sign(np.sign(u[1:-1,1:-1]) + 1)
	Mx1 = 1 - Mx2

	# Matrices where 1 is up, 0 is down or center
	My2 = np.sign(np.sign(v[1:-1,1:-1]) + 1)
	My1 = 1 - My2

	# Matrices of absolute values for u and v
	au = np.abs(u[1:-1,1:-1])
	av = np.abs(v[1:-1,1:-1]) 

	# Matrices of coefficients, respectively central, external, same x, same y
	ΔxΔy = Δx*Δy
	Cc = (Δx - au*Δt) * (Δy - av*Δt) / ΔxΔy
	Ce = Δt*Δt*au * av / ΔxΔy
	Cmx = (Δx - au*Δt) * av*Δt / ΔxΔy
	Cmy = Δt*au * (Δy - av*Δt) / ΔxΔy

	# Computes the advected quantity
	adv_q = np.empty_like(q)
	adv_q[1:-1,1:-1] = (
		  Cc * q[1:-1, 1:-1]
		+ Ce * ( Mx1*My1 * q[2: , 2: ] 
			   + Mx2*My1 * q[:-2, 2: ]
			   + Mx1*My2 * q[2: , :-2]
			   + Mx2*My2 * q[:-2, :-2] )
		+ Cmx * ( My1 * q[1:-1, 2:]
				+ My2 * q[1:-1, :-2] )
		+ Cmy * ( Mx1 * q[2:, 1:-1]
				+ Mx2 * q[:-2, 1:-1] )
	)
	return adv_q

def SemiLag2 (u,v, q, Δx,Δy, Δt):
	"""
	Second order semi-lagrangian 2D advection of the scalar field q by (u,v),
	based on SemiLag, with forward-backward error correction.
	Reslting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""
	qforw = SemiLag(u,v, q, Δx,Δy, Δt)
	qback = SemiLag(-u,-v, qforw, Δx,Δy, Δt)    
	# qcorr = q + (q-qback)/2
	adv_q = SemiLag(u,v, (3*q-qback)/2, Δx,Δy, Δt)
	return adv_q


########################## Differential operators ##########################

def Laplacian (f, Δx,Δy):
	"""
	Laplacian of the 2D scalar field f[i,j], which must contain ghost points.
	Finite differences (1,-2,1), order 2.
	Resulting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""
	Δx_2, Δy_2 = 1/(Δx*Δx), 1/(Δy*Δy)

	lap_f = np.empty_like(f)
	lap_f[1:-1,1:-1] = (
		  (f[2:  , 1:-1] + f[ :-2, 1:-1]) * Δx_2
		+ (f[1:-1, 2:  ] + f[1:-1,  :-2]) * Δy_2
		- 2 * f[1:-1, 1:-1] * (Δx_2+Δy_2)
	)
	return lap_f

def Divergence (u,v, Δx,Δy):
	"""
	Divergence of the 2D vector field (u,v)[i,j], which must contain ghost points.
	Centered finite differences (1,0,-1)/2.
	Reslting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""
	div = np.empty_like(u)
	div[1:-1,1:-1] = (
		  (u[2:, 1:-1] - u[:-2, 1:-1]) /Δx/2
		+ (v[1:-1, 2:] - v[1:-1, :-2]) /Δy/2
	)
	return div

def FD_1D_Laplacian_matrix (N_phys, Δx, BCdir_left, BCdir_right):
	"""
	1D Laplacian FD matrix. N_phys is the number of physical grid points, ie. /not/ including
	possible ghost points. `BCdir_left` and `BCdir_right` set boundary conditions to Dirichlet
	if True and Neumann if False, in a ghost-points-approach.
	Author : Emmanuel Dormy.
	"""
	
	# building the sparse matrix (tridiagonal 1,-2,1)
	diagonals = [np.ones(N_phys-1), -2*np.ones(N_phys), np.ones(N_phys-1)]
	diagonals[1][0]  = -3 if BCdir_left else -1.
	diagonals[1][-1] = -3 if BCdir_right else -1.
	offsets = np.array([-1,0,1])
	D2 = sp.diags(diagonals,offsets, shape=(N_phys,N_phys),format = 'csr') / Δx**2
	
	return D2


def FD_2D_Laplacian_matrix (Nx_phys, Ny_phys, Δx, Δy, BCdir_left=True, BCdir_right=True, BCdir_top=True, BCdir_bot=True):
	"""
	2D Laplacian FD matrix, Kronecker product of 1D matrices.
	See `FD_1D_Laplacian_matrix` for documentation.
	In CFD with fractional time stepping, typically used for pressure determination / solenoidal projection.
	Author : Emmanuel Dormy.
	"""

	DXX = FD_1D_Laplacian_matrix(Nx_phys, Δx, BCdir_left, BCdir_right)
	DYY = FD_1D_Laplacian_matrix(Ny_phys, Δy, BCdir_top, BCdir_bot)
	
	####### 2D Laplace operator
	LAP = sp.kron(DXX,sp.eye(Ny_phys,Ny_phys),format = 'csr') + sp.kron(sp.eye(Nx_phys,Nx_phys),DYY,format = 'csr')
	
	####### Correction matrix

	### Upper Diagonal terms
	dataNYNXi = [np.zeros(Ny_phys*Nx_phys)]
	offset = np.array([1])

	### Fix coef: 2+(-1) = 1 ==> Dirichlet at a single point
	# The value is set at one point (here [0][1]) to ensure uniqueness
	dataNYNXi[0][1] = -1 / Δx**2

	LAP0 = sp.diags(dataNYNXi,offset, shape=(Ny_phys*Nx_phys,Ny_phys*Nx_phys),format = 'csr')
  
	return LAP #+ LAP0
	#return LAP



def LUdecomposition(LAP):
	"""
	return the Incomplete LU decomposition 
	of a sparse matrix LAP
	"""
	return  lg.splu(LAP.tocsc(),)


def Resolve(splu,RHS):
	"""
	solve the system

	SPLU * x = RHS

	Args:
	--RHS: 2D array((NY,NX))
	--splu: (Incomplete) LU decomposed matrix 
			shape (NY*NX, NY*NX)

	Return: x = array[NY,NX]
	
	Rem1: taille matrice fonction des CL 

	"""
	# array 2D -> array 1D
	f2 = RHS.ravel()

	# Solving the linear system
	x = splu.solve(f2)

	return x.reshape(RHS.shape)
