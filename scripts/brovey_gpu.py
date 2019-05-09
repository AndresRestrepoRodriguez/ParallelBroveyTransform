# -*- coding: utf-8 -*-

print 'Importando librerias...'
import skimage.io
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
from pycuda.elementwise import ElementwiseKernel
import pycuda.cumath as cumath
import sys, getopt
import time

print 'Librerias importadas correctamente'

input_mul = sys.argv[1]
input_pan = sys.argv[2]
output_fus = sys.argv[3]

end = 0
start = 0

# Kernel establecido para llevar a cabo el método de rescale global min
lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *z",
        "z[i] = ((x[i]-a)*255)/(b-a)",
        "linear_combination")

# El step_1 realiza la la división de la banda entre la suma de las bandas
def step_1(matrix_color, matrix_suma):
    #La función gpuarray.if_positive evalua cada posición de la matriz
    #Y de acuerdo a su valor realiza la primer operación o la segunda constatando una sentencia If Else
    matrix_1 = gpuarray.if_positive(matrix_suma,(3*matrix_color)/matrix_suma,matrix_suma)
    return matrix_1

# El step_2 realiza la multiplicación posición a posición del step_1 por la pancromatica
def step_2(matrix_1, matrix_image_pan):
    #La función linalg.mulitply realiza la multiplicación elemento a elemento entre dos matrices
    matrix_2 = linalg.multiply(matrix_1, matrix_image_pan)
    return matrix_2

# El step_3 determina el valor maximo y minimo de la banda resultante de la transformada de brovey
def step_3(matrix_1):
    #La función np.max determina el maximo valor de un array
    #La función np.min determina el minimo valor de un array
    mat_max = np.amax(matrix_1.get())
    mat_min = np.amin(matrix_1.get())
    return mat_max, mat_min

# El step_4 realiza un ajuste de riqueza espectral llamado rescale global min
def step_4(matrix_1, matrix_color, mat_max, mat_min):
    #La función lin_comb invoca el kernel elementwise establecido para llevar a cabo el ajuste por rescale global min
    lin_comb(mat_min, matrix_1, mat_max, matrix_color)
    return matrix_color

# Lee la imagen multiespectral y la pancromatica
m = skimage.io.imread(input_mul, plugin='tifffile')
p = skimage.io.imread(input_pan, plugin='tifffile')

#Verifica que ambas imagenes cumplan con las condiciones
if m.shape[2]:
    print 'la imagen multiespectral tiene '+str(m.shape[2])+' bandas y tamaño '+str(m.shape[0])+'x'+str(m.shape[1])
else:
    print 'la primera imagen no es multiespectral'

if len(p.shape) == 2:
    print 'la imagen Pancromatica tiene tamaño '+str(p.shape[0])+'x'+str(p.shape[1])
else:
    print 'la segunda imagen no es pancromatica'

# Convierte a float32 y separa las bandas RGB de la multiespectral
m1 = m.astype(np.float32)
r = m[:,:,0]
r1 = r.astype(np.float32)
g = m[:,:,1]
g1 = g.astype(np.float32)
b = m[:,:,2]
b1 = b.astype(np.float32)

# Convierte la pancromatica a float32
p1 = p.astype(np.float32)

# Suma las bandas de la multiespectral
msuma = r1+g1+b1


start=time.time()
r1_gpu = gpuarray.to_gpu(r1)
g1_gpu = gpuarray.to_gpu(g1)
b1_gpu = gpuarray.to_gpu(b1)
p1_gpu = gpuarray.to_gpu(p1)
msuma_gpu = gpuarray.to_gpu(msuma)

linalg.init()
m11_gpu = step_1(r1_gpu, msuma_gpu)
m22_gpu = step_2(m11_gpu, p1_gpu)

m33_gpu = step_1(b1_gpu, msuma_gpu)
m44_gpu = step_2(m33_gpu, p1_gpu)

m55_gpu = step_1(g1_gpu, msuma_gpu)
m66_gpu = step_2(m55_gpu, p1_gpu)

Amax_host, Amin_host = step_3(m22_gpu)
rr_gpu = gpuarray.empty_like(r1_gpu)
step_4(m22_gpu, rr_gpu, Amax_host, Amin_host)

Amax_host, Amin_host = step_3(m66_gpu)
gg_gpu = gpuarray.empty_like(g1_gpu)
step_4(m66_gpu, gg_gpu, Amax_host, Amin_host)

Amax_host, Amin_host = step_3(m44_gpu)
bb_gpu = gpuarray.empty_like(b1_gpu)
step_4(m44_gpu, bb_gpu, Amax_host, Amin_host)
end = time.time()

ggg_host = gg_gpu.get().astype(np.uint8)
rrr_host = rr_gpu.get().astype(np.uint8)
bbb_host = bb_gpu.get().astype(np.uint8)

t = skimage.io.imsave('/home/nvera/andres/images/redBand8192.tif', rrr_host, plugin='tifffile')
t1 = skimage.io.imsave('/home/nvera/andres/images/greenBand8192.tif', ggg_host, plugin='tifffile')
t2 = skimage.io.imsave('/home/nvera/andres/images/blueBand8192.tif', bbb_host, plugin='tifffile')


# Combina las bandas resultantes
finalImage = np.stack((rrr_host, ggg_host, bbb_host))

# Guarda la imagen resultando de acuerdo al tercer parametro establecido en la linea de ejecución del script
t = skimage.io.imsave('/home/nvera/andres/images/'+output_fus+'.tif',finalImage, plugin='tifffile')


#Tiempo de ejecución para la transformada de Brovey en GPU
tiempo = (end-start)

print 'tiempo'
print tiempo
