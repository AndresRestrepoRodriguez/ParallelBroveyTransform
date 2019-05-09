# -*- coding: utf-8 -*-
import skimage.io
import numpy as np
import cv2
import sys, getopt
import time

input_mul = sys.argv[1]
input_pan = sys.argv[2]
output_fus = sys.argv[3]

end = 0
start = 0

# El step_1 realiza la la división de la banda entre la suma de las bandas
def step_1(matrix_1, matrix_color, msuma_matrix):
    for m in range(matrix_1.shape[0]):
        for n in range(matrix_1.shape[0]):
            if (msuma_matrix[m,n] != 0):
                matrix_1[m,n] = (3*matrix_color[m,n])/msuma_matrix[m,n]
            else:
                matrix_1[m,n] = msuma_matrix[m,n]
    return matrix_1

# El step_2 realiza la multiplicación posición a posición del step_1 por la pancromatica
def step_2(matrix_1, matrix_2, matrix_image_pan):
    for m in range(matrix_2.shape[0]):
        for n in range(matrix_2.shape[0]):
            matrix_2[m,n] = matrix_1[m,n]*matrix_image_pan[m,n]
    return matrix_2

# El step_3 determina el valor maximo y minimo de la banda resultante de la transformada de brovey
def step_3(matrix_1):
    mat_max = np.amax(matrix_1)
    mat_min = np.amin(matrix_1)
    return mat_max, mat_min

# El step_4 realiza un ajuste de riqueza espectral llamado rescale global min
def step_4(matrix_1, matrix_color, mat_max, mat_min):
    for m in range(matrix_color.shape[0]):
        for n in range(matrix_color.shape[0]):
            matrix_color[m,n] = (((matrix_1[m,n]-mat_min)*255)/(mat_max-mat_min))
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
m11 = np.zeros_like(r1)
m11 = step_1(m11, r1, msuma)

m22 = np.zeros_like(g1)
m22 = step_2(m11, m22, p1)

m33 = np.zeros_like(b1)
m33 = step_1(m33, b1, msuma)

m44 = np.zeros_like(b1)
m44 = step_2(m33, m44, p1)

m55 = np.zeros_like(g1)
m55 = step_1(m55, g1, msuma)

m66 = np.zeros_like(g1)
m66 = step_2(m55, m66, p1)


Amax, Amin = step_3(m22)
rr = np.zeros_like(r1)
rr = step_4(m22, rr, Amax, Amin)



Amax, Amin = step_3(m66)
gg = np.zeros_like(g1)
gg = step_4(m66, gg, Amax, Amin)



Amax, Amin = step_3(m44)
bb = np.zeros_like(b1)
bb = step_4(m44, bb, Amax, Amin)

end = time.time()

rrr = rr.astype(np.uint8)
ggg = gg.astype(np.uint8)
bbb = bb.astype(np.uint8)

# Combina las bandas resultantes
finalImage = np.stack((rrr_host, ggg_host, bbb_host))

# Guarda la imagen resultando de acuerdo al tercer parametro establecido en la linea de ejecución del script
t = skimage.io.imsave('/home/nvera/andres/images/'+output_fus+'.tif',finalImage, plugin='tifffile')

#Tiempo de ejecución para la transformada de Brovey en GPU
tiempo = (end-start)

print 'tiempo'
print tiempo
