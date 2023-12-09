
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
from skimage import feature

# Define una función para realizar una ventana deslizante (sliding window) sobre una imagen.
def sliding_window(img, 
                   patch_size,  # Define el tamaño del parche (patch) basado en el primer parche positivo por defecto
                   istep=2,  # Paso de desplazamiento en la dirección i (verticalmente)
                   jstep=2,  # Paso de desplazamiento en la dirección j (horizontalmente)
                   scale=1.0):  # Factor de escala para ajustar el tamaño del parche
                   
    # Calcula las dimensiones Ni y Nj del parche ajustadas por el factor de escala.
    Ni, Nj = (int(scale * s) for s in patch_size)
    
    # Itera a lo largo de la imagen en la dirección i
    for i in range(0, img.shape[0] - Ni, istep):
        # Itera a lo largo de la imagen en la dirección j
        for j in range(0, img.shape[1] - Ni, jstep):
            
            # Extrae el parche de la imagen usando las coordenadas actuales i, j.
            patch = img[i:i + Ni, j:j + Nj]
            
            # Si el factor de escala es diferente de 1, redimensiona el parche al tamaño original del parche.
            if scale != 1:
                patch = resize(patch, patch_size)
            
            # Usa yield para devolver las coordenadas actuales y el parche. 
            # Esto convierte la función en un generador.
            yield (i, j), patch


def non_max_suppression(indices, Ni, Nj, overlapThresh):
    # Si no hay rectángulos, regresar una lista vacía
    if len(indices) == 0:
        return []

    # Si las cajas son enteros, convertir a flotantes
    if indices.dtype.kind == "i":
        indices = indices.astype("float")

    # Inicializar la lista de índices seleccionados
    pick = []

    # Tomar las coordenadas de los cuadros
    x1 = np.array([indices[i,0] for i in range(indices.shape[0])])
    y1 = np.array([indices[i,1] for i in range(indices.shape[0])])
    x2 = np.array([indices[i,0]+Ni for i in range(indices.shape[0])])
    y2 = np.array([indices[i,1]+Nj for i in range(indices.shape[0])])

    # Calcula el área de los cuadros y ordena los cuadros
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Mientras todavía hay índices en la lista de índices
    while len(idxs) > 0:
        # Toma el último índice de la lista y agrega el índice a la lista de seleccionados
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Encontrar las coordenadas (x, y) más grandes para el inicio de la caja y las coordenadas (x, y) más pequeñas para el final de la caja
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Calcula el ancho y alto de la caja
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Calcula la proporción de superposición
        overlap = (w * h) / area[idxs[:last]]

        # Elimina todos los índices del índice de lista que tienen una proporción de superposición mayor que el umbral proporcionado
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Devuelve solo las cajas seleccionadas
    return indices[pick].astype("int")

# Función que devuelve el número de detecciones brutas y procesadas para diversas escalas
# Esta función asume conocidos model, size y los parámetros de las HOG
def detections_by_scale(model, test_image, test_scales, step, positive_shape, thresholds=[0.5]):
    raw_detections = []
    detections = []
        
    for scale in tqdm(test_scales):
        raw_detections_scale = []
        detections_scale = []

        # Ventana deslizante
        indices, patches = zip(*sliding_window(test_image, patch_size=positive_shape,scale=scale, istep=step, jstep=step))

        # Calcula las características HOG para cada parche y las almacena en un array.
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        # Predicción
        for thr in thresholds:
            labels = (model.predict_proba(patches_hog)[:,1]>=thr).astype(int)
            raw_detections_scale.append(labels.sum())
            Ni, Nj = positive_shape
            indices = np.array(indices)
            detecciones = indices[labels == 1]
            detecciones = non_max_suppression(np.array(detecciones),Ni,Nj, 0.3)
            detections_scale.append(len(detecciones))
        
        # Actualizamos las listas
        raw_detections.append(raw_detections_scale)
        detections.append(detections_scale)
        
    return np.array(raw_detections), np.array(detections)