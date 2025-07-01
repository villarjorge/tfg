import numpy as np
import matplotlib.pyplot as plt

def construir_b(b, rho, dt, u, v, w, dx, dy, dz):
	b[1:-1, 1:-1, 1:-1] = rho * (
		(1/dt)*(
			(0.5/dx)*(u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 0:-2]) 
			+ (0.5/dy)*(v[1:-1, 2:, 1:-1] - v[1:-1, 0:-2, 1:-1])
			+ (0.5/dz)*(w[2:, 1:-1, 1:-1] - w[0:-2, 1:-1, 1:-1])
		) 
		- ((0.5/dx)*(u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 0:-2]))**2 
		- ((0.5/dy)*(v[1:-1, 2:, 1:-1] - v[1:-1, 0:-2, 1:-1]))**2
		- ((0.5/dz)*(w[2:, 1:-1, 1:-1] - w[0:-2, 1:-1, 1:-1]))**2
		- ((1/dx)*(u[1:-1, 2:, 1:-1] - u[1:-1, 0:-2, 1:-1])*(1/dy)*(v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, 0:-2]))
		- ((1/dx)*(u[2:, 1:-1, 1:-1] - u[0:-2, 1:-1, 1:-1])*(1/dy)*(w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 0:-2]))
		- ((1/dx)*(v[2:, 1:-1, 1:-1] - v[0:-2, 1:-1, 1:-1])*(1/dy)*(w[1:-1, 2:, 1:-1] - w[1:-1, 0:-2, 1:-1]))
	)
	return b

def poisson_presion(p, dx, dy, dz, b):
	pn = np.empty_like(p)
	pn = p.copy()
	
	for _ in range(nit):
		pn = p.copy()
		p[1:-1, 1:-1, 1:-1] = (0.5/(dx**2 + dy**2 + dz**2))*(
			(
				(pn[1:-1, 1:-1, 2:] + pn[1:-1, 1:-1, 0:-2])*dy**2*dz**2
				+ (pn[1:-1, 2:, 1:-1] + pn[1:-1, 0:-2, 1:-1])*dx**2*dz**2
				+ (pn[2:, 1:-1, 1:-1] + pn[0:-2, 1:-1, 1:-1])*dx**2*dy**2
			)
			- dx**2*dy**2*dz**2*b[1:-1, 1:-1, 1:-1]
		)
		# Condiciones de contorno para la presión
		p[:, :, -1] = p[:, :, -2] # dp/dx = 0 en x = 2
		p[:, :, 0] = p[:, :, 1]   # dp/dx = 0 en x = 0
		p[:, 0, :] = p[:, 1, :]   # dp/dy = 0 en y = 0
		p[0, :, :] = p[1, :, :]   # dp/dz = 0 en z = 0

		p[:, -1, :] = 0        # p = 0 en y = 2
		p[-1, :, :] = 0        # p = 0 en z = 2
		
	return p

def simulacion(nt, u, v, w, dt, dx, dy, dz, p, rho, nu):
	un = np.empty_like(u)
	vn = np.empty_like(v)
	wn = np.empty_like(v)
	b = np.zeros((ny, nx, nz))

	U = list()
	V = list()
	W = list()
	P = list()

	for n in range(nt):
		un = u.copy()
		vn = v.copy()

		b = construir_b(b, rho, dt, u, v, w, dx, dy, dz)
		p = poisson_presion(p, dx, dy, dz, b)
		P.append(p)

		u[1:-1, 1:-1, 1:-1] = (
			un[1:-1, 1:-1, 1:-1] 
			- un[1:-1, 1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 0:-2]) 
			- vn[1:-1, 1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 0:-2, 1:-1]) 
			- wn[1:-1, 1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1, 1:-1] - un[0:-2, 1:-1, 1:-1])
			- dt/(2*rho*dx) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, 0:-2]) 
			+ nu*(
				dt / dx**2 * (un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2]) 
				+ dt / dy**2 * (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 0:-2, 1:-1])
				+ dt / dz**2 * (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[0:-2, 1:-1, 1:-1])
			)
		)

		v[1:-1, 1:-1, 1:-1] = (
			vn[1:-1, 1:-1, 1:-1] 
			- un[1:-1, 1:-1, 1:-1,] * dt/dx * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, 0:-2])
			- vn[1:-1, 1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 0:-2, 1:-1]) 
			- wn[1:-1, 1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1, 1:-1] - vn[0:-2, 1:-1, 1:-1])
			- dt/(2*rho*dy) * (p[1:-1, 2:, 1:-1] - p[1:-1, 0:-2, 1:-1]) 
			+ nu*(
				dt / dx**2 * (vn[1:-1, 1:-1, 2:] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 1:-1, 0:-2]) 
				+ dt / dy**2 * (vn[1:-1, 2:, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 0:-2, 1:-1])
				+ dt / dz**2 * (vn[2:, 1:-1, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[0:-2, 1:-1, 1:-1])
			)
		)

		w[1:-1, 1:-1, 1:-1] = (
			wn[1:-1, 1:-1, 1:-1] 
			- un[1:-1, 1:-1, 1:-1,] * dt/dx * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, 0:-2])
			- vn[1:-1, 1:-1, 1:-1] * dt/dy * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 0:-2, 1:-1]) 
			- wn[1:-1, 1:-1, 1:-1] * dt/dy * (wn[1:-1, 1:-1, 1:-1] - wn[0:-2, 1:-1, 1:-1])
			- dt/(2*rho*dy) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, 0:-2]) 
			+ nu*(
				dt / dx**2 * (wn[1:-1, 1:-1, 2:] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 1:-1, 0:-2]) 
				+ dt / dy**2 * (wn[1:-1, 2:, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 0:-2, 1:-1])
				+ dt / dz**2 * (wn[2:, 1:-1, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[0:-2, 1:-1, 1:-1])
			)
		)
		# Condiciones de contorno
		u[0, :, :]  = 0
		u[:, 0, :]  = 0
		u[:, :, 0]  = 0
		u[-1, :, :] = 0
		u[:, -1, :] = 0
		u[:, :, -1] = 0

		v[0, :, :]  = 0
		v[:, 0, :]  = 0
		v[:, :, 0]  = 0
		v[-1, :, :] = 0
		v[:, -1, :] = 0
		v[:, :, -1] = 0

		w[0, :, :]  = 0
		w[:, 0, :]  = 0
		w[:, :, 0]  = 1
		w[-1, :, :] = 0
		w[:, -1, :] = 0
		w[:, :, -1] = 0

		U.append(u)
		V.append(v)
		W.append(w)

	return u, v, w, p, U, V, W, P

def simulacion_visualizacion(nt, u, v, w, dt, dx, dy, dz, p, rho, nu):
	un = np.empty_like(u)
	vn = np.empty_like(v)
	wn = np.empty_like(v)
	b = np.zeros_like(u)

	# Setup plot for visualization
	ax = plt.figure(figsize=(12, 14)).add_subplot(projection='3d')

	# Make the grid
	x = np.linspace(xmin, xmax, nx)
	y = np.linspace(ymin, ymax, ny)
	z = np.linspace(zmin, zmax, nz)

	X, Y, Z = np.meshgrid(x, y, z)

	tomar_cada = 4

	ax.quiver(
        X[::tomar_cada, ::tomar_cada, ::tomar_cada], 
        Y[::tomar_cada, ::tomar_cada, ::tomar_cada], 
        Z[::tomar_cada, ::tomar_cada, ::tomar_cada], 
        u[::tomar_cada, ::tomar_cada, ::tomar_cada], 
        v[::tomar_cada, ::tomar_cada, ::tomar_cada], 
        w[::tomar_cada, ::tomar_cada, ::tomar_cada], 
        length=0.1
	)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.ion() # Turn on interactive mode
	plt.show()

	for n in range(nt):
		un = u.copy()
		vn = v.copy()

		b = construir_b(b, rho, dt, u, v, w, dx, dy, dz)
		p = poisson_presion(p, dx, dy, dz, b)

		u[1:-1, 1:-1, 1:-1] = (
			un[1:-1, 1:-1, 1:-1] 
			- un[1:-1, 1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, 0:-2]) 
			- vn[1:-1, 1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 0:-2, 1:-1]) 
			- wn[1:-1, 1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1, 1:-1] - un[0:-2, 1:-1, 1:-1])
			- dt/(2*rho*dx) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, 0:-2]) 
			+ nu*(
				dt / dx**2 * (un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, 0:-2]) 
				+ dt / dy**2 * (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 0:-2, 1:-1])
				+ dt / dz**2 * (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[0:-2, 1:-1, 1:-1])
			)
		)

		v[1:-1, 1:-1, 1:-1] = (
			vn[1:-1, 1:-1, 1:-1] 
			- un[1:-1, 1:-1, 1:-1,] * dt/dx * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, 0:-2])
			- vn[1:-1, 1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 0:-2, 1:-1]) 
			- wn[1:-1, 1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1, 1:-1] - vn[0:-2, 1:-1, 1:-1])
			- dt/(2*rho*dy) * (p[1:-1, 2:, 1:-1] - p[1:-1, 0:-2, 1:-1]) 
			+ nu*(
				dt / dx**2 * (vn[1:-1, 1:-1, 2:] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 1:-1, 0:-2]) 
				+ dt / dy**2 * (vn[1:-1, 2:, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 0:-2, 1:-1])
				+ dt / dz**2 * (vn[2:, 1:-1, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[0:-2, 1:-1, 1:-1])
			)
		)

		w[1:-1, 1:-1, 1:-1] = (
			wn[1:-1, 1:-1, 1:-1] 
			- un[1:-1, 1:-1, 1:-1,] * dt/dx * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, 0:-2])
			- vn[1:-1, 1:-1, 1:-1] * dt/dy * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 0:-2, 1:-1]) 
			- wn[1:-1, 1:-1, 1:-1] * dt/dy * (wn[1:-1, 1:-1, 1:-1] - wn[0:-2, 1:-1, 1:-1])
			- dt/(2*rho*dy) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, 0:-2]) 
			+ nu*(
				dt / dx**2 * (wn[1:-1, 1:-1, 2:] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 1:-1, 0:-2]) 
				+ dt / dy**2 * (wn[1:-1, 2:, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 0:-2, 1:-1])
				+ dt / dz**2 * (wn[2:, 1:-1, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[0:-2, 1:-1, 1:-1])
			)
		)
		# Condiciones de contorno
		u[0, :, :]  = 0
		u[:, 0, :]  = 0
		u[:, :, 0]  = 0
		u[-1, :, :] = 0
		u[:, -1, :] = 1
		u[:, :, -1]  = 0

		v[0, :, :]  = 0
		v[:, 0, :]  = 0
		v[:, :, 0]  = 0
		v[-1, :, :] = 0
		v[:, -1, :] = 0
		v[:, :, -1]  = 0

		w[0, :, :]  = 0
		w[:, 0, :]  = 0
		w[:, :, 0]  = 0
		w[-1, :, :] = 0
		w[:, -1, :] = 0
		w[:, :, -1]  = 0

		# visualization
		if n%10 == 0:
			ax.quiver(
				X[::tomar_cada, ::tomar_cada, ::tomar_cada], 
				Y[::tomar_cada, ::tomar_cada, ::tomar_cada], 
				Z[::tomar_cada, ::tomar_cada, ::tomar_cada], 
				u[::tomar_cada, ::tomar_cada, ::tomar_cada], 
				v[::tomar_cada, ::tomar_cada, ::tomar_cada], 
				w[::tomar_cada, ::tomar_cada, ::tomar_cada], 
				length=0.1
			)
			plt.pause(0.01) # Pause to allow plot to update

	return u, v, w, p


def representar_campo_vectorial(x, y, z, u, v, w):
	# https://matplotlib.org/stable/gallery/mplot3d/quiver3d.html#sphx-glr-gallery-mplot3d-quiver3d-py
	ax = plt.figure().add_subplot(projection='3d')
	X, Y, Z = np.meshgrid(x, y, z)

	tomar_cada = 4

	#ax.quiver(X, Y, Z, u, v, w, length=0.1, normalize=True)

	quiver_plot = ax.quiver(
		X[::tomar_cada, ::tomar_cada, ::tomar_cada], 
		Y[::tomar_cada, ::tomar_cada, ::tomar_cada], 
		Z[::tomar_cada, ::tomar_cada, ::tomar_cada], 
		u[::tomar_cada, ::tomar_cada, ::tomar_cada], 
		v[::tomar_cada, ::tomar_cada, ::tomar_cada], 
		w[::tomar_cada, ::tomar_cada, ::tomar_cada], 
		length=0.1
	) # , normalize=True

	plt.xlabel('X')
	plt.ylabel('Y')

	plt.show()
	#print("Quiver plot: ", type(quiver_plot))

from matplotlib.animation import FuncAnimation

def animar_campo_vectorial(x, y, z, U_data, V_data, W_data):
	"""
	Crea una animación de un campo vectorial 3D que cambia con el tiempo.

	Args:
		x, y, z (np.array): Arrays 1D que definen las coordenadas de la rejilla.
		U_data, V_data, W_data (list of np.array): Listas de arrays 3D. 
													Cada elemento de la lista representa 
													los componentes u, v, w del campo 
													vectorial en un instante de tiempo.
													Todos los arrays deben tener la misma forma.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	X, Y, Z = np.meshgrid(x, y, z)

	tomar_cada = 4

	# Inicializar el quiver con el primer fotograma de datos
	quiver_plot = ax.quiver(
		X[::tomar_cada, ::tomar_cada, ::tomar_cada],
		Y[::tomar_cada, ::tomar_cada, ::tomar_cada],
		Z[::tomar_cada, ::tomar_cada, ::tomar_cada],
		U_data[0][::tomar_cada, ::tomar_cada, ::tomar_cada],
		V_data[0][::tomar_cada, ::tomar_cada, ::tomar_cada],
		W_data[0][::tomar_cada, ::tomar_cada, ::tomar_cada],
		length=0.1
	)

	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z') # Añadir etiqueta para el eje Z

	def update(frame):
		"""
		Función de actualización para la animación.
		Actualiza los vectores del quiver para cada fotograma.
		"""
		# Actualizamos los datos del quiver_plot
		# quiver_plot = ax.quiver(
		# 	X[::tomar_cada, ::tomar_cada, ::tomar_cada],
		# 	Y[::tomar_cada, ::tomar_cada, ::tomar_cada],
		# 	Z[::tomar_cada, ::tomar_cada, ::tomar_cada],
		# 	U_data[frame][::tomar_cada, ::tomar_cada, ::tomar_cada],
		# 	V_data[frame][::tomar_cada, ::tomar_cada, ::tomar_cada],
		# 	W_data[frame][::tomar_cada, ::tomar_cada, ::tomar_cada],
		# 	length=0.1
		# )
		quiver_plot.set_UVWC(
			U_data[frame][::tomar_cada, ::tomar_cada, ::tomar_cada], 
			V_data[frame][::tomar_cada, ::tomar_cada, ::tomar_cada],
			W_data[frame][::tomar_cada, ::tomar_cada, ::tomar_cada]
		)
		return quiver_plot,

	# Crear la animación
	ani = FuncAnimation(fig, update, frames=len(U_data), blit=True, interval=50) # interval en ms
	#ani.save("anim.mp4", fps=20)

	plt.show()

if __name__ == "__main__":
	# Variables
	# Resouluciones espaciotemporales
	print("iniciando")
	nx = 41
	ny = 41
	nz = 41
	nt = 500
	# Pseudo tiempo para resolver la ecuación de Poisson
	nit = 50

	xmin = 0
	xmax = 2
	ymin = 0
	ymax = 2
	zmin = 0
	zmax = 2

	dx = (xmax - xmin)/(nx - 1) # Distancia entre puntos en la x
	dy = (ymax - ymin)/(ny - 1) # Distancia entre puntos en la y
	dz = (zmax - zmin)/(nz - 1) # Distancia entre puntos en la y

	rho = 1 
	nu = 0.1
	dt = 0.001

	# Make the grid
	x = np.linspace(xmin, xmax, nx)
	y = np.linspace(ymin, ymax, ny)
	z = np.linspace(zmin, zmax, nz)

	#u = np.zeros((ny, nx, nz))
	#v = np.zeros((ny, nx, nz))
	#w = np.zeros((ny, nx, nz))

	# Inicializar velocidades pequeñas aleatorias. Distribución normal mu = 0, sigma = 0.1
	mu = 0
	sigma = 0.2
	u = np.random.normal(mu, sigma, (ny, nx, nz))
	v = np.random.normal(mu, sigma, (ny, nx, nz))
	w = np.random.normal(mu, sigma, (ny, nx, nz))

	print("Estado inicial")
	representar_campo_vectorial(x, y, z, u, v, w)

	p = np.zeros((ny, nx, nz))
	b = np.zeros((ny, nx, nz))
	print("Computando")
	u, v, w, p, U, V, W, P = simulacion(nt, u, v, w, dt, dx, dy, dz, p, rho, nu)
	#u, v, w, p = simulacion_visualizacion(nt, u, v, w, dt, dx, dy, dz, p, rho, nu)
	print("Representando")

	representar_campo_vectorial(x, y, z, u, v, w)
	#animar_campo_vectorial(x, y, z, U, V, W)