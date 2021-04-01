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

########################## Spectral methods routines ##########################


def FFT_2D_phys2spec (field, NXSp,NYSp):
	"""
	Fourier transform from 2D physical space to spectral space.
	Author : Emmanuel Dormy.
	"""
	Nx,Ny = field.shape
	# Total nb of points
	NN = Nx*Ny
	# Fourier transform
	ft_field_temp = fft.fft2( field / NN )
	# Only keep the relevant modes
	NXSp_2 = int(NXSp/2)
	NYSp_2 = int(NYSp/2)
	ft_field = np.zeros((NXSp,NYSp), dtype=complex)
	ft_field[0:NXSp_2   , 0:NYSp_2   ] = ft_field_temp[0:NXSp_2    , 0:NYSp_2    ]
	ft_field[NXSp_2:NXSp, 0:NYSp_2   ] = ft_field_temp[Nx-NXSp_2:Nx, 0:NYSp_2    ]
	ft_field[0:NXSp_2   , NYSp_2:NYSp] = ft_field_temp[0:NXSp_2    , Ny-NYSp_2:Ny]
	ft_field[NXSp_2:NXSp, NYSp_2:NYSp] = ft_field_temp[Nx-NXSp_2:Nx, Ny-NYSp_2:Ny]
	return ft_field

def FFT_2D_spec2phys (ft_field, Nx,Ny):
	"""
	Fourier transform from spectral space to 2D physical space.
	Author : Emmanuel Dormy.
	"""
	NXSp,NYSp = ft_field.shape
	# Total nb of points
	NN = Nx*Ny
	# Temporary array to FFT, filled with relevant modes
	ft_field_temp = np.zeros((Nx,Ny), dtype=complex)
	NXSp_2 = int(NXSp/2)
	NYSp_2 = int(NYSp/2)
	ft_field_temp[0:NXSp_2    , 0:NYSp_2    ] = ft_field[0:NXSp_2   , 0:NYSp_2   ]
	ft_field_temp[Nx-NXSp_2:Nx, 0:NYSp_2    ] = ft_field[NXSp_2:NXSp, 0:NYSp_2   ]
	ft_field_temp[0:NXSp_2    , Ny-NYSp_2:Ny] = ft_field[0:NXSp_2   , NYSp_2:NYSp]
	ft_field_temp[Nx-NXSp_2:Nx, Ny-NYSp_2:Ny] = ft_field[NXSp_2:NXSp, NYSp_2:NYSp]
	# Inverse Fourier transform
	field = NN * np.real( fft.ifft2(ft_field_temp) )
	return field


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
	Reslting ghost points are undefined and have to be set afterwards.
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
	diagonals = [np.ones(N_phys), -2*np.ones(N_phys), np.ones(N_phys)]
	diagonals[1][0]  = -3 if BCdir_left else -1.
	diagonals[1][-1] = -3 if BCdir_right else -1.
	offsets = np.array([-1,0,1])
	D2 = sp.dia_matrix((diagonals,offsets), shape=(N_phys,N_phys)) / Δx**2
	
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
	LAP = sp.kron(DXX,sp.eye(Ny_phys,Ny_phys)) + sp.kron(sp.eye(Nx_phys,Nx_phys),DYY)
	
	####### Correction matrix

	### Upper Diagonal terms
	dataNYNXi = [np.zeros(Ny_phys*Nx_phys)]
	offset = np.array([1])

	### Fix coef: 2+(-1) = 1 ==> Dirichlet at a single point
	# The value is set at one point (here [0][1]) to ensure uniqueness
	dataNYNXi[0][1] = -1 / Δx**2

	LAP0 = sp.dia_matrix((dataNYNXi,offset), shape=(Ny_phys*Nx_phys,Ny_phys*Nx_phys))
  
	return LAP + LAP0
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

##### Tentative perso :
## Newton pour l'implicite :
## Update rho first:
def adv_scal(u,v,x,Δx,Δy):
    return 1/2/Δx * u*(x[1:,:]-x[:-1,:])[1:-1,1:-1] + 1/2/Δy * v*(x[:,1:]-x[:,:-1])[1:-1,1:-1]

def update_rho(u,v,ρ,Δx,Δy,Δt):
    F = np.zeros_like(ρ)
    DF = np.zeros_like(ρ)
    Err = 2*ϵ
    while Err>ϵ:
        ## Get F:
        F[1:-1,1:-1] = 1/Δt * (x-ρ)[1:-1,1:-1] + adv_scal(u,v,x,Δx,Δy)
        # Top
 
        ## Get DF:
        DF[1:-1,1:-1] = 1/Δt * x[1:-1,1:-1] + adv_scal(u,v,x,Δx,Δy)

        ## Newton's method:
        d = np.linalg.solve(DF,-F)
        x = x + d
        Err = np.linalg.norm(d, ord = inf)
    return x
## Approximate u^{n+1}:
def approx_u(u,v,ρ,ρ_,s,fx,fy,Δx,Δy,Δt,LAP):
    X = u
    Y = v
    Nabsx = 1/2/Δx * (s[1:,:]-s[:-1,:])[1:-1,1:-1]
    Nabsy = 1/2/Δy * (s[:,1:]-s[:,:-1])[1:-1,1:-1]
    Cx = -ρ_ * 1/Δt * X + 1/Re * Nabsx - fx
    Cy = -ρ_ * 1/Δt * Y + 1/Re * Nabsy - fy
    Fx,Fy = np.zeros_like(X),np.zeros_like(Y)
    DFx, DFy = np.zeros_like(X),np.zeros_like(Y)
    Err = 2*ϵ
    while Err>ϵ:
        ## Get F:
        Lapu = np.dot(LAP,X)[1:-1,1:-1]
        Lapv = np.dot(LAP,Y)[1:-1,1:-1]
        X_, Y_ = X[1:-1,1:-1], Y[1:-1,1:-1]
        Fx = ρ_ * 1/Δt * X_ + ρ*adv_scal(u,v,X,Δx,Δy)[1:-1,1:-1] - 1/Re * Lapu + C
        Fy = ρ_ * 1/Δt * Y_ + ρ*adv_scal(u,v,Y,Δx,Δy)[1:-1,1:-1] - 1/Re * Lapv + C
        DFx = ρ_ * 1/Δt * X_ + ρ*adv_scal(u,v,X,Δx,Δy)[1:-1,1:-1] - 1/Re * Lapu
        DFy = ρ_ * 1/Δt * Y_ + ρ*adv_scal(u,v,Y,Δx,Δy)[1:-1,1:-1] - 1/Re * Lapu
        ## Newton's method:
        dx = np.linalg.solve(DFx,-Fx)
        dy = np.linalg.solve(DFy,-Fy)
        X = X + dx
        Y = Y + dy
        Err = np.max(np.linalg.norm(dx, ord = inf),np.linalg.norm(dy, ord = inf))
        ## Conditions limites :
    return X,Y

## Find phi: Peut-être résolu liénairement ??
def update_phi(phi,u,v,ρ,Δx,Δy):
    x = phi
    
## First step: get u_star:
def get_non_lin_term(u,v,ρ,Δx,Δy):
    #Attention : GhostPoints?
    nlx = Divergence(ρ*u*u,ρ*u*v,Δx,Δy)
    nly = Divergence(ρ*v*u,ρ*v*v,Δx,Δy)
    return nlx, nly
    
def update_grad_p(press,dx_press,dy_press,Δx,Δy):
    dx_press[1:-1, :] = (press[2:, :] - press[:-2, :])/Δx/2
    dy_press[:, 1:-1] = (press[:, 2:] - press[:, :-2])/Δy/2
    
def get_uv_star(u,v,β,press,dx_press,dy_press,ρ,fu,fv,Δx,Δy,Δt):
    nlu, nlv = get_non_lin_term(u,v,ρ,Δx,Δy)
    Lap_u = np.dot(LAP,u)
    Lap_v = np.dot(LAP,v)
    u_star = u - Δt/ρ * nlu - β*Δt/ρ * dx_press + Δt/Re * Lap_u + Δt*fu # Attention : ne marche que pour un champ de force type gravitée!! sinon diviser f par rho...
    v_star = v - Δt/ρ * nlv - β*Δt/ρ * dy_press + Δt/Re * Lap_v + Δt*fv
    return u_star, v_star
    
## Second step: update rho:

    
## Third step: projection step (div un+1 = 0):          
### Opérateurs sous forme matricielle :
# Retourne la matrice correspondant à (X.Nablda). Ici sous version 2D, prend Nx, Ny la taille de la grille
def grad(s,Δx,Δy):
    dX = 1/Δx/2 * (s[1:,:]-s[:-1,:])[1:-1,1:-1]
    dY = 1/Δy/2 * (s[:,1:]-s[:,:-1])[1:-1,1:-1]
    return dX, dY
def advect(u,v,Nx,Ny):
    dNX = [1/Δx/2 * u*np.ones(NXx), np.zeros(Nx), -1/Δx/2 * u*np.ones(Nx)]
    dNX = [1/Δy/2*v*np.ones(Ny), np.zeros(Ny), -1/Δy/2*v*np.ones(Ny)]##Il manque un truc,non ? C'est pas des np.ones qu'il faut mais un autre truc
    return sp.kron(sp.eye(Ny,Ny), dNx) + sp.kron(dNy, sp.eye(Nx,Nx))
def get_rho(u,v,ρ,Nx,Ny,Δx,Δy,Δt):
    A = LUdecomposition(np.eye(Nx*Ny) +Δt*advect(u,v,Nx,Ny))
    ## Attention aux conditions limites !
    return Resolve(A,ρ)
def approx_u(u,v,ρ,ρ_,s,fx,fy,Δx,Δy,Δt,LAP):
    A = np.eye(Nx*Ny) + ρ*Δt/ρ_ * advect(u,v,Nx,Ny) - Δt/Re * LAP
    LU = LUdecomposition(A)
    Nabsx,Nabsy = grad(s,Δx,Δy)
    u_hatx = Resolve(LU,Δt*f-Δt/Re * Nabsx + u)
    u_haty = Resolve(LU,Δt*f-Δt/Re * Nabsy + u)
    ## Attention aux conditions limites !
    return u_hatx,u_haty

def get_phi(ρ,u_hatx,u_haty,Δx,Δy,Nx,Ny):
    LapPart = 1/ρ*LAP
    Δx2 = Δx*Δx*4
    Δy2 = Δy*Δy*4
    dataNx = [np.ones(Nx), np.zeros(Nx), -np.ones(Nx)]
    dataNy = [np.ones(Ny), np.zeros(Ny), -np.ones(Ny)]
    offsets = np.array([-1,0,1]) 
    Dx = sp.dia_matrix((dataNx,offsets), shape=(Nx,Nx)) ## Matrice de dérivée selon x## Matrice de dérivée selon x
    Dy = sp.dia_matrix((dataNy,offsets), shape=(Ny,Ny)) ## Matrice de dérivée selon x## Matrice de dérivée selon y
    Dx = sp.kron(sp.eye(Ny,Ny), Dx)
    Dy = sp.kron(Dy, sp.eye(Nx,Nx))
    Mat = 1/Δx2 * (ρ**-1[1:,:]-ρ**-1[:-1,:])[1:-1,1:-1]*Dx + 1/Δy2 * (ρ**-1[:,1:]-ρ**-1[:,:-1])[1:-1,1:-1]*Dy + LapPart
    RHS = - Divergence(u_hatx,u_haty, Δx,Δy)
    
    phi = Resolve(Mat, RHS)
    return phi

def update_u_s(u,v,s,ρ,phi,u_hatx,u_haty,Δx,Δy):
    dX,dY = grad(phi,Δx,Δy)
    u = u_hatx + 1/ρ * dX
    v = u_haty + 1/ρ * dY
    s = s - Divergence(u_hatx,u_haty, Δx,Δy)