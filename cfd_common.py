import numpy as np
import scipy.fftpack as fft
import matplotlib
import matplotlib.pyplot as plt
import scipy
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

def SemiLag (u,v, q, ??x,??y, ??t):
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
	??x??y = ??x*??y
	Cc = (??x - au*??t) * (??y - av*??t) / ??x??y
	Ce = ??t*??t*au * av / ??x??y
	Cmx = (??x - au*??t) * av*??t / ??x??y
	Cmy = ??t*au * (??y - av*??t) / ??x??y

	# Computes the advected quantity
	adv_q = np.zeros_like(q)
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

def SemiLag2 (u,v, q, ??x,??y, ??t, coeff=1):
	"""
	Second order semi-lagrangian 2D advection of the scalar field q by (u,v),
	based on SemiLag, with forward-backward error correction.
	Reslting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""
	qforw = SemiLag(u,v, q, ??x,??y, ??t)
	qback = SemiLag(-u,-v, qforw, ??x,??y, ??t)    
	qcorr = coeff*(q-qback)/2
	qcorr[ 0, :] = 0
	qcorr[-1, :] = 0
	qcorr[:,  0] = 0
	qcorr[:, -1] = 0
	qcorr[ 1, :] = 0
	qcorr[-2, :] = 0
	qcorr[:,  1] = 0
	qcorr[:, -2] = 0
	adv_q = SemiLag(u,v, q+qcorr, ??x,??y, ??t)
	return adv_q


########################## Differential operators ##########################

def Laplacian (f, ??x,??y):
	"""
	Laplacian of the 2D scalar field f[i,j], which must contain ghost points.
	Finite differences (1,-2,1), order 2.
	Reslting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""
	??x_2, ??y_2 = 1/(??x*??x), 1/(??y*??y)

	lap_f = np.empty_like(f)
	lap_f[1:-1,1:-1] = (
		  (f[2:  , 1:-1] + f[ :-2, 1:-1]) * ??x_2
		+ (f[1:-1, 2:  ] + f[1:-1,  :-2]) * ??y_2
		- 2 * f[1:-1, 1:-1] * (??x_2+??y_2)
	)
	return lap_f

def Divergence (u,v, ??x,??y):
	"""
	Divergence of the 2D vector field (u,v)[i,j], which must contain ghost points.
	Centered finite differences (1,0,-1)/2.
	Reslting ghost points are undefined and have to be set afterwards.
	Author : Emmanuel Dormy.
	"""
	div = np.empty_like(u)
	div[1:-1,1:-1] = (
		  (u[2:, 1:-1] - u[:-2, 1:-1]) /??x/2
		+ (v[1:-1, 2:] - v[1:-1, :-2]) /??y/2
	)
	return div


def FD_1D_Laplacian_matrix (N_phys, ??x, BCdir_left, BCdir_right):
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
	D2 = sp.dia_matrix((diagonals,offsets), shape=(N_phys,N_phys)) / ??x**2
	
	return D2


def FD_2D_Laplacian_matrix (Nx_phys, Ny_phys, ??x, ??y, BCdir_left=True, BCdir_right=True, BCdir_top=True, BCdir_bot=True):
	"""
	2D Laplacian FD matrix, Kronecker product of 1D matrices.
	See `FD_1D_Laplacian_matrix` for documentation.
	In CFD with fractional time stepping, typically used for pressure determination / solenoidal projection.
	Author : Emmanuel Dormy.
	"""

	DXX = FD_1D_Laplacian_matrix(Nx_phys, ??x, BCdir_left, BCdir_right)
	DYY = FD_1D_Laplacian_matrix(Ny_phys, ??y, BCdir_bot, BCdir_top)
	
	####### 2D Laplace operator
	LAP = sp.kron(DXX,sp.eye(Ny_phys,Ny_phys)) + sp.kron(sp.eye(Nx_phys,Nx_phys),DYY)

	# LU decomposition
	LU_decomp = lg.splu(LAP.tocsc(),)

	# Solver function LAP.X = RHS
	def LAP_solver (RHS, BCdir_val_left=None, BCdir_val_right=None, BCdir_val_top=None, BCdir_val_bot=None):
		# include Dirichlet boundary conditions in the RHS
		RHS_BC = np.copy(RHS)
		if BCdir_val_left is not None:
			if not BCdir_left: raise ValueError("Can't set left boundary value if the matrix is not constructed with left Dirichlet BC")
			RHS_BC[0,:] += -2 * BCdir_val_left / ??x**2
		if BCdir_val_right is not None:
			if not BCdir_right: raise ValueError("Can't set right boundary value if the matrix is not constructed with right Dirichlet BC")
			RHS_BC[-1,:] += -2 * BCdir_val_right / ??x**2
		if BCdir_val_top is not None:
			if not BCdir_top: raise ValueError("Can't set top boundary value if the matrix is not constructed with top Dirichlet BC")
			RHS_BC[:,0] += -2 * BCdir_val_top / ??y**2
		if BCdir_val_bot is not None:
			if not BCdir_bot: raise ValueError("Can't set bottom boundary value if the matrix is not constructed with bottom Dirichlet BC")
			RHS_BC[:,-1] += -2 * BCdir_val_bot / ??y**2
		# flatten
		f2 = RHS_BC.ravel()
		# solve
		X = LU_decomp.solve(f2)
		return X.reshape(RHS.shape)

	return LAP, LAP_solver


def FD_2D_D??D_matrix (Nx_phys, Ny_phys, ??x, ??y, ??, BCdir_left=True, BCdir_right=True, BCdir_top=True, BCdir_bot=True, backend_build_C=False):
	"""
	2D ?????(?????) FD matrix, where ?? is a matrix containing ghost points.
	Reduces to the laplacian matrix FD_2D_Laplacian_matrix is ?? is a constant field.
	"""

	if backend_build_C:
		from cfdutils import ffi, lib as cfdutils_lib
		row = np.zeros(5*Nx_phys*Ny_phys, dtype=np.uint32)
		col = np.zeros(5*Nx_phys*Ny_phys, dtype=np.uint32)
		val = np.zeros(5*Nx_phys*Ny_phys, dtype=np.float64)
		cfdutils_lib.build_FD_2D_DrhoD_matrix(Nx_phys, Ny_phys, ??x, ??y, ffi.cast("double*",??.ctypes.data), BCdir_left, BCdir_right, BCdir_top, BCdir_bot, ffi.cast("uint32_t*",row.ctypes.data), ffi.cast("uint32_t*",col.ctypes.data), ffi.cast("double*",val.ctypes.data))
	else:
		??x2 = ??x**2
		??y2 = ??y**2
		row = np.zeros(5*Nx_phys*Ny_phys, dtype=int)
		col = np.zeros(5*Nx_phys*Ny_phys, dtype=int)
		val = np.zeros(5*Nx_phys*Ny_phys)
		N = Ny_phys
		k = 0
		for i in range(0,Nx_phys):
			for j in range(0,Ny_phys):
				c = i*N+j
				vcenter = 0

				if j < Ny_phys-1:
					row[k] = i*N+(j+1)
					col[k] = c
					val[k] = 1/??[i+1,j+1] / ??y2
					k += 1
					vcenter += -1/??[i+1,j+1] / ??y2   # ??0
				elif BCdir_bot:
					vcenter += -2/??[i+1,j+1] / ??y2

				if j > 0:
					row[k] = i*N+(j-1)
					col[k] = c
					val[k] = 1/??[i+1,j] / ??y2
					k += 1
					vcenter += -1/??[i+1,j] / ??y2   # ??-
				elif BCdir_top:
					vcenter += -2/??[i+1,j] / ??y2

				if i < Nx_phys-1:
					row[k] = (i+1)*N+j
					col[k] = c
					val[k] = 1/??[i+1,j+1] / ??x2
					k += 1
					vcenter += -1/??[i+1,j+1] / ??x2   # ??0
				elif BCdir_right:
					vcenter += -2/??[i+1,j+1] / ??x2

				if i > 0:
					row[k] = (i-1)*N+j
					col[k] = c
					val[k] = 1/??[i,j+1] / ??x2
					k += 1
					vcenter += -1/??[i,j+1] / ??x2   # ??-
				elif BCdir_left:
					vcenter += -2/??[i,j+1] / ??x2

				row[k] = i*N+j
				col[k] = c
				val[k] = vcenter
				k += 1

	D??D = sp.csc_matrix((val,(row,col)), shape=(Nx_phys*Ny_phys,Nx_phys*Ny_phys))
	LU_decomp = lg.splu(D??D,)

	# Solver function D??D.X = RHS
	def D??D_solver (RHS, BCdir_val_left=None, BCdir_val_right=None, BCdir_val_top=None, BCdir_val_bot=None):
		# include Dirichlet boundary conditions in the RHS
		RHS_BC = np.copy(RHS)
		if BCdir_val_left is not None:
			if not BCdir_left: raise ValueError("Can't set left boundary value if the matrix is not constructed with left Dirichlet BC")
			RHS_BC[0,:] += -2 * BCdir_val_left / ??x**2 / ??[0,1:-1]
		if BCdir_val_right is not None:
			if not BCdir_right: raise ValueError("Can't set right boundary value if the matrix is not constructed with right Dirichlet BC")
			RHS_BC[-1,:] += -2 * BCdir_val_right / ??x**2 / ??[-2,1:-1]
		if BCdir_val_top is not None:
			if not BCdir_top: raise ValueError("Can't set top boundary value if the matrix is not constructed with top Dirichlet BC")
			RHS_BC[:,0] += -2 * BCdir_val_top / ??y**2 / ??[1:-1,0]
		if BCdir_val_bot is not None:
			if not BCdir_bot: raise ValueError("Can't set bottom boundary value if the matrix is not constructed with bottom Dirichlet BC")
			RHS_BC[:,-1] += -2 * BCdir_val_bot / ??y**2 / ??[1:-1,-2]
		# flatten
		f2 = RHS_BC.ravel()
		# solve
		X = LU_decomp.solve(f2)
#		X = lg.spsolve(D??D, f2)
		return X.reshape(RHS.shape)

	return D??D, D??D_solver


def FD_2D_ILapAdv_matrix (Nx_phys, Ny_phys, ??x, ??y, ??t, ??, ??, ??_next, u, v, BCdir_left=True, BCdir_right=True, BCdir_top=True, BCdir_bot=True):

	from cfdutils import ffi, lib as cfdutils_lib
	row = np.zeros(5*Nx_phys*Ny_phys, dtype=np.uint32)
	col = np.zeros(5*Nx_phys*Ny_phys, dtype=np.uint32)
	val = np.zeros(5*Nx_phys*Ny_phys, dtype=np.float64)
	cfdutils_lib.build_FD_2D_ILapAdv_centered_matrix(Nx_phys, Ny_phys, ??x, ??y, ffi.cast("double*",??.ctypes.data), ffi.cast("double*",??_next.ctypes.data), ffi.cast("double*",u.ctypes.data), ffi.cast("double*",v.ctypes.data), ??, ??t, BCdir_left, BCdir_right, BCdir_top, BCdir_bot, ffi.cast("uint32_t*",row.ctypes.data), ffi.cast("uint32_t*",col.ctypes.data), ffi.cast("double*",val.ctypes.data))
	
	ILapAdv = sp.csc_matrix((val,(row,col)), shape=(Nx_phys*Ny_phys,Nx_phys*Ny_phys))
	LU_decomp = lg.splu(ILapAdv,)

	# Solver function A.X = RHS
	def ILapAdv_solver (RHS, BCdir_val_left=None, BCdir_val_right=None, BCdir_val_top=None, BCdir_val_bot=None):
		# include Dirichlet boundary conditions in the RHS
		RHS_BC = np.copy(RHS)
		if BCdir_val_left is not None:
			if not BCdir_left: raise ValueError("Can't set left boundary value if the matrix is not constructed with left Dirichlet BC")
			RHS_BC[0,:] += BCdir_val_left * ( 2*?? / ??x**2 + ??_next[1,1:-1] * u[1,1:-1] / ??x )
		if BCdir_val_right is not None:
			if not BCdir_right: raise ValueError("Can't set right boundary value if the matrix is not constructed with right Dirichlet BC")
			RHS_BC[-1,:] += BCdir_val_right * ( 2*?? / ??x**2 - ??_next[-2,1:-1] * u[-2,1:-1] / ??x )
		if BCdir_val_top is not None:
			if not BCdir_top: raise ValueError("Can't set top boundary value if the matrix is not constructed with top Dirichlet BC")
			RHS_BC[:,0] += BCdir_val_top * ( 2*?? / ??y**2 - ??_next[1:-1,1] * v[1:-1,1] / ??y )
		if BCdir_val_bot is not None:
			if not BCdir_bot: raise ValueError("Can't set bottom boundary value if the matrix is not constructed with bottom Dirichlet BC")
			RHS_BC[:,-1] += BCdir_val_bot * ( 2*?? / ??y**2 + ??_next[1:-1,-2] * v[1:-1,-2] / ??y )
		# flatten
		f2 = RHS_BC.ravel()
		# solve
		X = LU_decomp.solve(f2)
		return X.reshape(RHS.shape)

	return ILapAdv, ILapAdv_solver
