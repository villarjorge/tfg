# -*- coding: utf-8 -*-

# Contiene las funciones de simulación usadas en otros sitios

import fdtd
import numpy as np
import scipy as sc
from scipy.ndimage import zoom

from ciddor_air import n

def permitivity_ciddor_one(longitud_onda:float, T, h):
    """
    Usa la función dada por Cidor para calcular la permitividad a partir de la temperatura y la humedad, asumiendo que el aire se comporta como un gas idealç
    No tiene en cuenta los cambios de la presión según la humedad
    """
    R_esp = 287.052874 # Constante de gas ideal específica del aire J kg-1 K-1
    rho_aire = 1.225 # kg/m3
    pressure = rho_aire*R_esp*(273 + T) # Simple aproximación, no tiene en cuenta los cambios de humedad 

    return n(longitud_onda*1e6, T, pressure, h, 350)**2 # Recuerda que longitud de onda es en micrometros

def permitivity_sasiela(longitud_onda:float, T, P):
    """
    Usa una estimación dada en Sasiela para encontrar la variación de la permitividad a partir del indice de refracción
    longitud_onda: longitud de onda en um (1e-6 m)
    T: Temperatura en kelvin
    P: presión en milibares (100 pa)
    """
    C_1 = 77.6e-6
    C_2 = 7.52e-3

    index = 1 + C_1*(1 + C_2/longitud_onda**2)*(P/T)

    return index**2

def set_up_grid(longitud_onda:float, shape=(400, 200, 1), ancho_fuente=0.5,  perm=1.0005441500841912, patch=None):
    """
    Inicializa la grid de FDTD con un valor constante de permitividad dado. Opcionalmente coloca una sección de aire con otra permitividad.
    La grid creada tiene como condiciones de contorno PLM (perfectly matched layer), que son una forma de implementar condiciones de contorno absorbentes. 
    Además la grid tiene una fuente con la longitud de onda dada y un detector
    Ancho fuente: ancho de la fuente como fracción del dominio
    """
    # para longitud de onda: debe ser en  metros -> el spacing debe ser 10 veces menor que la longitud de onda para tener estabilidad
    grid = fdtd.Grid(
        shape = shape, # Dando el último valor como uno se crea una grid de dos dimensiones
        permittivity=perm,
        grid_spacing=longitud_onda*0.1
    )
    # Si se da un patch, ponerlo en la grid
    if patch is not None:
        tamaño_x, tamaño_y = patch.shape[:2]
        # Offsets para que esté centrado en x e y
        offset_x = (grid.shape[0] - tamaño_x)//2
        offset_y = (grid.shape[1] - tamaño_y)//2
        grid[offset_x:offset_x+tamaño_x, offset_y:offset_y+tamaño_y, 0] = fdtd.Object(permittivity=patch, name="Aire turbulento")

    # Añadimos una fuente con perfil gaussiano dado el tamaño, centrada en la mitad de x
    tamaño_y = int(grid.shape[1]*ancho_fuente)
    offset_y = (grid.shape[1] - tamaño_y)//2
    # tamaño_y = grid.shape[1]*1//10
    # offset_y = grid.shape[1]*9//20

    grid[grid.shape[0]-40, offset_y:offset_y+tamaño_y, 0] = fdtd.LineSource(
        period = longitud_onda / (3e8), name="Fuente", amplitude=2.
    ) # 20:80
    # Añadimos un detector
    grid[20, :, 0] = fdtd.LineDetector(name="detector")

    # Añade PLM en los contornos
    # x
    grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
    grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

    # y
    grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
    grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")
    
    return grid

def create_patch_temperature_ciddor(longitud_onda:float, shape=(400, 200, 1), mean_T=25, std_T=1, mean_h=0.2, std_h=0.01, kernel=np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])):
    """
    Crea un patch donde la permitividad es un campo aleatorio para pasarlo a la función que genera la grid. 
    Esta versión obtiene la permitividad a partir de la temperatura y la humedad fraccionaria, que se modelan como campos aleatorios. 
    Opcionalmente estos campos se pueden convolucionar con un kernel dado para suavizar los mismos
    """
    # Tamaño en x: fracción del dominio
    tamaño_x = shape[0]*2//3
    # Tamaño en y: fracción del dominio
    tamaño_y = shape[1]*4//6
    # Temperaturas en grados centigrados
    T_patch = np.random.normal(mean_T, std_T, (tamaño_x, tamaño_y, 1))
    # Humedad fraccional de 0 a 1
    h_patch = np.random.normal(mean_h, std_h, (tamaño_x, tamaño_y, 1))

    print("T", np.mean(T_patch), np.std(T_patch), T_patch.shape)
    print("H", np.mean(h_patch), np.std(h_patch), h_patch.shape)

    # Realiza una convolución para suavizar los campos
    T_patch = sc.signal.convolve(T_patch, kernel, mode="same") # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html
    h_patch = sc.signal.convolve(h_patch, kernel, mode="same")
    print("T", np.mean(T_patch), np.std(T_patch), T_patch.shape)
    print("H", np.mean(h_patch), np.std(h_patch), h_patch.shape)

    patch = permitivity_ciddor_one(longitud_onda, T_patch, h_patch)
    print("n", np.mean(patch), np.std(patch), patch.shape)
    return patch

def create_patch_permitivity(shape=(400, 200, 1), final_shape=(400, 200, 1), mean_perm=1+1.5e-4, std_perm=1.72e-5, kernel=np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])):
    """
    Crea una zona donde la permitividad es un campo aleatorio. Inicialmente se crea con dimensiones dadas en shape y luego se escala para llegar a las dimensiones de final_shape. 
    """
    # Crea el patch y realiza una interpolación lineal para llegar a la forma deseada
    patch = np.random.normal(mean_perm, std_perm, shape)
    zoomed_patch = zoom(patch, zoom=[final_shape[i]/shape[i] for i in range(0, 3)])

    # Realiza una convolución para suavizar los campos
    zoomed_patch = sc.signal.convolve(zoomed_patch, kernel, mode="same") # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html

    return zoomed_patch

def get_average_detector_data(grid, t_index_1, t_index_2, kind="E"):
    """
    Toma el promedio de E^2 (no el cuadrado del promedio) de lo que llega al detector de la grid.
    El promedio es lo que detecta entre los tiempos t_index_1, t_index_2. Debes dar los tiempos como indices
    """
    valores = grid.detector.detector_values()[kind]
    valores = np.array(valores)

    tol = 1e-10
    valores = np.where(np.abs(valores) > tol, valores, 0.0)

    magnitud_detector_E1 = np.einsum("ijk,ijk->ij", valores, valores)[t_index_1:t_index_2]

    return np.average(magnitud_detector_E1, axis=0)

def get_amplitude_detector_data(grid, kind="E"):
    """
    Obten la magnitud de la amplitud de la onda, simplemente encontrando el máximo de la magnitud al cuadrado y haciendo su raiz cuadrada
    """
    valores = grid.detector.detector_values()[kind]
    valores = np.array(valores)

    tol = 1e-10
    valores = np.where(np.abs(valores) > tol, valores, 0.0)

    magnitud_detector_E1 = np.einsum("ijk,ijk->ij", valores, valores)

    return np.sqrt(magnitud_detector_E1.max())

def gaussian_kernel_nd(shape, sigma):
    """
    Crea un kernel gausiano de n dimensiones
    
    Parámetros:
    shape: tuple de ints - forma del kernel (Ejemplos (5, 5) para 2D, (5, 5, 5) para 3D)
    sigma: float o tuple - desviaciones estandar para cada dimensión
    """
    # Create a delta function at the center
    kernel = np.zeros(shape)
    center = tuple(s // 2 for s in shape)
    kernel[center] = 1.0
    
    # Apply Gaussian filter to create the kernel
    return sc.ndimage.gaussian_filter(kernel, sigma=sigma)