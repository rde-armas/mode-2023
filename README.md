
::: {.cell .markdown id="n63en9DiIsCL"}
# 1. Procesamiento y etiquetado de fondos {#1-procesamiento-y-etiquetado-de-fondos}
:::

::: {.cell .markdown id="qTU4tQSnIsCM"}
Vamos a trabajar con imagenes en escala de grises, cada una con
dimensiones de 62 x 47 pıxeles. Para asegurar coherencia, procesaremos
las fotografıas de fondo para que compartan estas propiedades.

En total, contamos con 13.233 imagenes de rostros, de las cuales
utilizaremos un 90 % para el entrenamiento, dejandonos aproximadamente
11.909 imagenes de rostros para dicho proposito
:::

::: {.cell .markdown id="GZ86mw5zIsCP"}
Con el objetivo de hacer que nuestro conjunto de datos refleje de manera
mas precisa la realidad, donde se requerir a detectar rostros en
imagenes donde la mayorıa de los parches seran fondos,hemos decidido que
por cada rostro en el conjunto de entrenamiento existan 5 fondos,
mientras que en el conjunto de prueba este numero se incrementa a 100.
:::

::: {.cell .code execution_count="37" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":400}" id="mImM5XDSIsCN" outputId="79162b9e-c94c-4c17-a427-206017ac0e1b"}
``` python
import code.carga  as carga

X_train, X_test, y_train, y_test = carga.get_train_test(5, 100)

print(f'muestras entrenamiento: {X_train.shape}')
print(f'muestras validacion: {X_test.shape}')
```

::: {.output .stream .stdout}
    muestras entrenamiento: (71357, 62, 47)
    muestras validacion: (133521, 62, 47)
:::
:::

::: {.cell .markdown id="rWrcmtrSxEiL"}
## 2. Features HOG {#2-features-hog}
:::

::: {.cell .code execution_count="38" id="BZl6yRIyxJx1"}
``` python
from skimage import feature, data, color
import matplotlib.pyplot as plt
import numpy as np
```
:::

::: {.cell .code execution_count="39" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":251}" id="Ag6i8kuNx8hp" outputId="56113f55-f125-432d-fb9c-5cff1adc9bdb"}
``` python
muestra_ejemplo = carga.positive_patches()[58]
hog_features, hog_vis = feature.hog(muestra_ejemplo, visualize=True)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(muestra_ejemplo, cmap='gray')
ax[0].set_title('Imagen de entrada')

ax[1].imshow(hog_vis, cmap='gray')
ax[1].set_title('Visualización de las HOG features')
```

::: {.output .execute_result execution_count="39"}
    Text(0.5, 1.0, 'Visualización de las HOG features')
:::

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/293a291514861ad3ed3b95be322b177adea83cad.png)
:::
:::

::: {.cell .markdown id="OCxvUTjF6CG3"}
## 3. Comparacion de modelos {#3-comparacion-de-modelos}
:::

::: {.cell .markdown id="oRWXU8M70kEz"}
Entre todos los experimentos realizados, seleccionamos los 10 modelos
más destacados, principalmente basándonos en sus métricas de Balanced
Accuracy y FPR (tasa de falsos positivos).

Dado que nuestra meta es maximizar la detección de caras en un modelo
desbalanceado, utilizamos la Balanced Accuracy y, al mismo tiempo,
buscamos minimizar los falsos positivos (FPR).
:::

::: {.cell .code execution_count="40" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="CislGnJV0kE1" outputId="ec22f71b-865b-4ba9-e96d-6f3a0f52bc8c"}
``` python
import pandas as pd
import tabulate

path = './results/results.csv'
df = pd.read_csv(path)

# Ordenar por FPR y TPR por separado
df_ba = df.sort_values(by=['Balanced Accuracy'], ascending=False).head(100)
df_fpr = df_ba.sort_values(by=['FPR'], ascending=True).head(10)

results = []

# Iterar sobre los clasificadores en df_fpr y buscar coincidencias en df_tpr
for _, row in df_fpr.iterrows():

    # Agregar a la lista de resultados
    results.append([
        row['Classifier'],
        row['Preprocessing'],
        row['Accuracy'],
        row['Precision'],
        row['Recall/TPR'],
        row['FPR'],
        row['FNR'],
        row['TNR'],
        row['F1-Score'],
        row['ROC curve (area)'],
        row['Balanced Accuracy'],
        row['Time Train']
    ])

# Imprimir la tabla usando tabulate
headers = ["Classifier", "Preprocessing", "Accuracy", "Precision", "Recall/TPR", "FPR", "FNR", "TNR", "F1-Score", "ROC curve (area)", "Balanced Accuracy", "Time Train"]
print(tabulate.tabulate(results, headers, tablefmt="grid"))
```

::: {.output .stream .stdout}
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | Classifier                                                                                                        | Preprocessing   |   Accuracy |   Precision |   Recall/TPR |       FPR |         FNR |      TNR |   F1-Score |   ROC curve (area) |   Balanced Accuracy |   Time Train |
    +===================================================================================================================+=================+============+=============+==============+===========+=============+==========+============+====================+=====================+==============+
    | GBoostingClassifier: n_estimators: 100, loss: exponential, max_depth: 5, min_samples_split: 2, max_features: sqrt | HOGPrepocess    |   0.999453 |    0.968539 |     0.999765 | 0.0314607 | 0.000234518 | 0.968539 |   0.972546 |           0.988134 |            0.988134 |     107.922  |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: exponential, max_depth: 5, min_samples_split: 2, max_features: log2 | HOGPrepocess    |   0.999371 |    0.962687 |     0.999743 | 0.0373134 | 0.000257223 | 0.962687 |   0.968468 |           0.986971 |            0.986971 |      32.8228 |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: log_loss, max_depth: 5, min_samples_split: 5, max_features: sqrt    | HOGPrepocess    |   0.999431 |    0.962222 |     0.999811 | 0.0377778 | 0.000189149 | 0.962222 |   0.971578 |           0.990366 |            0.990366 |     107.813  |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: exponential, max_depth: 5, min_samples_split: 5, max_features: sqrt | HOGPrepocess    |   0.999416 |    0.962166 |     0.999796 | 0.0378338 | 0.000204278 | 0.962166 |   0.970808 |           0.989611 |            0.989611 |     107.838  |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: log_loss, max_depth: 5, min_samples_split: 5, max_features: log2    | HOGPrepocess    |   0.999371 |    0.961997 |     0.99975  | 0.038003  | 0.000249661 | 0.961997 |   0.968492 |           0.987345 |            0.987345 |      32.7216 |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: exponential, max_depth: 5, min_samples_split: 3, max_features: log2 | HOGPrepocess    |   0.999326 |    0.960448 |     0.99972  | 0.0395522 | 0.000279919 | 0.960448 |   0.966216 |           0.985827 |            0.985827 |      32.8648 |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: log_loss, max_depth: 5, min_samples_split: 2, max_features: sqrt    | HOGPrepocess    |   0.999438 |    0.960206 |     0.999841 | 0.0397937 | 0.000158893 | 0.960206 |   0.972025 |           0.991865 |            0.991865 |     107.383  |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: log_loss, max_depth: 5, min_samples_split: 3, max_features: sqrt    | HOGPrepocess    |   0.999393 |    0.958672 |     0.999811 | 0.0413284 | 0.000189156 | 0.958672 |   0.969765 |           0.990347 |            0.990347 |     107.509  |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 100, loss: exponential, max_depth: 5, min_samples_split: 5, max_features: log2 | HOGPrepocess    |   0.999274 |    0.957494 |     0.999697 | 0.0425056 | 0.000302618 | 0.957494 |   0.963602 |           0.984679 |            0.984679 |      32.9266 |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
    | GBoostingClassifier: n_estimators: 50, loss: log_loss, max_depth: 5, min_samples_split: 5, max_features: sqrt     | HOGPrepocess    |   0.999101 |    0.949925 |     0.999599 | 0.0500747 | 0.000400959 | 0.949925 |   0.954921 |           0.979731 |            0.979731 |      54.0507 |
    +-------------------------------------------------------------------------------------------------------------------+-----------------+------------+-------------+--------------+-----------+-------------+----------+------------+--------------------+---------------------+--------------+
:::
:::

::: {.cell .markdown}
## 4. Entrenamiento y evaluación del modelo {#4-entrenamiento-y-evaluación-del-modelo}
:::

::: {.cell .code execution_count="41"}
``` python
from sklearn.ensemble import  GradientBoostingClassifier
import code.features as feat
from joblib import load, dump

# Entrenamos el modelo
#n_estimators: 100, loss: exponential, max_depth: 5, min_samples_split: 2, max_features: sqrt  
dt = GradientBoostingClassifier(n_estimators=100, loss='exponential', max_depth=5, min_samples_split=2, max_features='sqrt') 

X_train_prep = feat.HOGPrepocess().preprocess_imgs(X_train)
dt.fit(X_train_prep, y_train)
dump(dt, "./model/model.joblib") 
```

::: {.output .stream .stderr}
    Preprocessing Images HOGPrepocess: 100%|██████████| 71357/71357 [00:11<00:00, 6292.21image/s]
:::

::: {.output .execute_result execution_count="41"}
    ['./model/model.joblib']
:::
:::

::: {.cell .code execution_count="42"}
``` python
model_saved = load('./model/model.joblib')
X_test_prep = feat.HOGPrepocess().preprocess_imgs(X_test)
y_pred = model_saved.predict(X_test_prep)
y_pred_proba = model_saved.predict_proba(X_test_prep)[:,1]
```

::: {.output .stream .stderr}
    Preprocessing Images HOGPrepocess: 100%|██████████| 133521/133521 [00:19<00:00, 6716.02image/s]
:::
:::

::: {.cell .code execution_count="43"}
``` python
from sklearn.metrics import roc_curve

# Curva ROC y umbral óptimo
fig, ax = plt.subplots(1,2,figsize=(8, 8))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
gmean = np.sqrt(tpr * (1 - fpr))
index = np.argmax(gmean)
thresholdOpt = round(thresholds[index], ndigits = 4)
fprOpt = round(fpr[index], ndigits = 4)
tprOpt = round(tpr[index], ndigits = 4)

ax[0].step(
    fpr,
    tpr,
    lw=1,
    alpha=1,
)

ax[0].plot(
    fprOpt,
    tprOpt,
    marker = 'o'
)

ax[0].set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Curva ROC",
)
ax[0].axis("square")

ax[1].set_aspect('equal')
ax[1].set_xlim([-0.05, 0.1])
ax[1].set_xbound(lower=-0.05, upper=0.1)
ax[1].set_ylim([0.85,1])
ax[1].set_ybound(lower=0.86, upper=1.01)

ax[1].step(
    fpr,
    tpr,
    lw=1,
    alpha=1,
)

ax[1].plot(
    fprOpt,
    tprOpt,
    marker = 'o'
)

ax[1].set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Zoom",
)

plt.tight_layout()
plt.show()

print(f'Umbral óptimo: {thresholdOpt}')
print(f'FPR: {fprOpt}, TPR: {tprOpt}')
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/b950e06ccb46c14b0c0b69e4d89a0488dfa3e69f.png)
:::

::: {.output .stream .stdout}
    Umbral óptimo: 0.0433
    FPR: 0.0049, TPR: 0.9917
:::
:::

::: {.cell .code execution_count="44"}
``` python
# Otra forma de calcular un umbral adecuado
indx = np.argmax(tpr>=0.95)
thresholdAde = thresholds[indx]
print('Umbral adecuado: ', thresholdAde)
```

::: {.output .stream .stdout}
    Umbral adecuado:  0.7781326505100885
:::
:::

::: {.cell .markdown}
### 4.1 Test en la imagen del astronauta {#41-test-en-la-imagen-del-astronauta}
:::

::: {.cell .code execution_count="45"}
``` python
from skimage.transform import  rescale

# Imagen de prueba
test_image = data.astronaut()
test_image = color.rgb2gray(test_image)
test_image = rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]
```
:::

::: {.cell .code execution_count="46"}
``` python
# Visualizamos la imagen
# Buscamos la escala de los rostros
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')

# Defino las dimensiones del parche
positive_patche_shape = carga.positive_patches()[0].shape
true_scale = 1
Ni, Nj = positive_patche_shape

ax.add_patch(plt.Rectangle((0, 0), Nj, Ni, edgecolor='red', alpha=1, lw=1, facecolor='none'))
plt.show()
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/16a04c98ab0ceab26405890a54323b7375b82c06.png)
:::
:::

::: {.cell .code execution_count="47"}
``` python
import code.face_detector as detector

# Utiliza la función de ventana deslizante en una imagen de prueba.
indices, patches = zip(*detector.sliding_window(test_image, positive_patche_shape ,scale=true_scale))


# Calcula las características HOG para cada parche y las almacena en un array.
patches_hog = np.array([feature.hog(patch) for patch in patches])

# Muestra la forma del array de características HOG.
patches_hog.shape
```

::: {.output .execute_result execution_count="47"}
    (1911, 1215)
:::
:::

::: {.cell .markdown}
#### 4.1.1 Desempeño según umbrales {#411-desempeño-según-umbrales}
:::

::: {.cell .code execution_count="48"}
``` python
# Escalas a testear
test_scales = np.linspace(0.125, 2, 50)
```
:::

::: {.cell .code execution_count="49"}
``` python
# Predicción

# Umbral default
labels_default = model_saved.predict(patches_hog).astype(int)

# Umbral óptimo
labels_optimo = (model_saved.predict_proba(patches_hog)[:,1]>=thresholdOpt).astype(int)

# Umbral adecuado
labels_adecuado = (model_saved.predict_proba(patches_hog)[:,1]>=thresholdAde).astype(int)
```
:::

::: {.cell .code execution_count="50"}
``` python
Ni, Nj = positive_patche_shape
indices = np.array(indices)

# Umbral default
detecciones_default = indices[labels_default == 1]
detecciones_default = detector.non_max_suppression(np.array(detecciones_default),Ni,Nj, 0.3)

# Umbral optimo
detecciones_optimo = indices[labels_optimo == 1]
detecciones_optimo = detector.non_max_suppression(np.array(detecciones_optimo),Ni,Nj, 0.3)

# Umbral adecuado
detecciones_adecuado = indices[labels_adecuado == 1]
detecciones_adecuado = detector.non_max_suppression(np.array(detecciones_adecuado),Ni,Nj, 0.3)
```
:::

::: {.cell .code execution_count="51"}
``` python
# Visualizamos las detecciones
fig, ax = plt.subplots(1,3, figsize=(8,4))

# Umbral default
ax[0].imshow(test_image, cmap='gray')
ax[0].axis('off')

for i, j in detecciones_default:
    ax[0].add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',alpha=1, lw=1, facecolor='none'))

ax[0].set_title('Default')

# Umbral óptimo
ax[1].imshow(test_image, cmap='gray')
ax[1].axis('off')

for i, j in detecciones_optimo:
    ax[1].add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',alpha=1, lw=1, facecolor='none'))

ax[1].set_title('Óptimo')

# Umbral adecuado
ax[2].imshow(test_image, cmap='gray')
ax[2].axis('off')

for i, j in detecciones_adecuado:
    ax[2].add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',alpha=1, lw=1, facecolor='none'))

ax[2].set_title('Adecuado')

plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/7dc15189eb0084e31a8abafb1e44d931c6573742.png)
:::
:::

::: {.cell .markdown}
### 4.2 Desempeño en varias escalas según umbral {#42-desempeño-en-varias-escalas-según-umbral}
:::

::: {.cell .code execution_count="52"}
``` python
# Escalas a testear
test_scales = np.linspace(0.125, 2, 50)
```
:::

::: {.cell .code execution_count="53"}
``` python
raw_detections, detections = detector.detections_by_scale(
    model_saved,
    test_image,
    test_scales,
    positive_shape=positive_patche_shape,
    step=2,
    thresholds=[0.5, thresholdOpt, thresholdAde]
    )
```

::: {.output .stream .stderr}
    100%|██████████| 50/50 [00:53<00:00,  1.07s/it]
:::
:::

::: {.cell .code execution_count="54"}
``` python
number_faces = 1

fig, ax = plt.subplots(1,2, figsize=(12,4))

ax[0].set_title('Bruto')
ax[0].axvline(x=true_scale, ls = '--', color='red')
ax[0].step(test_scales, raw_detections[:,0], label = 'Default')
ax[0].step(test_scales, raw_detections[:,1], label = 'Óptimo')
ax[0].step(test_scales, raw_detections[:,2], label = 'Adecuado')
ax[0].grid(True)
ax[0].set_xlabel('Escalas')
ax[0].set_ylabel('Detecciones')
ax[0].legend()

ax[1].set_title('Procesado')
ax[1].axvline(x=true_scale, ls = '--', color='red')
ax[1].axhline(y=number_faces, ls = '--', color='red')
ax[1].step(test_scales, detections[:,0], label = 'Default')
ax[1].step(test_scales, detections[:,1], label = 'Óptimo')
ax[1].step(test_scales, detections[:,2], label = 'Adecuado')
ax[1].grid(True)
ax[1].set_xlabel('Escalas')
ax[1].set_ylabel('Detecciones')
ax[1].legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/0195ef144ac43bc70fb30de18a9426fb08d1f4cb.png)
:::
:::

::: {.cell .markdown}
## 5. Pruebas {#5-pruebas}
:::

::: {.cell .markdown}
### 5.1 Test con 2 rostros {#51-test-con-2-rostros}
:::

::: {.cell .code execution_count="55"}
``` python
test_image = plt.imread('./imagenes/2_Rostros.jpg')
test_image = color.rgb2gray(test_image)
test_image = rescale(test_image,0.5)
test_image.shape
```

::: {.output .execute_result execution_count="55"}
    (336, 501)
:::
:::

::: {.cell .code execution_count="56"}
``` python
# Visualizamos la imagen
# Buscamos la escala de los rostros
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')

scale = 1.6
Ni, Nj = positive_patche_shape

ax.add_patch(plt.Rectangle((0, 0), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
plt.show()
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/b7df26d1746a31d590cddbde45352279207699da.png)
:::
:::

::: {.cell .code execution_count="57"}
``` python
# Utiliza la función de ventana deslizante en una imagen de prueba.
# zip(*...) toma las tuplas generadas y las descompone en índices y parches.
indices, patches = zip(*detector.sliding_window(test_image, positive_patche_shape,scale=scale))

# Calcula las características HOG para cada parche y las almacena en un array.
patches_hog = np.array([feature.hog(patch) for patch in patches])

# Muestra la forma del array de características HOG.
patches_hog.shape
```

::: {.output .execute_result execution_count="57"}
    (23919, 1215)
:::
:::

::: {.cell .code execution_count="58"}
``` python
# Predicción
labels = (model_saved.predict_proba(patches_hog)[:,1]>=thresholdAde).astype(int)
labels.sum()
```

::: {.output .execute_result execution_count="58"}
    88
:::
:::

::: {.cell .code execution_count="59"}
``` python
Ni, Nj = (int(scale*s) for s in positive_patche_shape)
indices = np.array(indices)
detecciones = indices[labels == 1]
detecciones = detector.non_max_suppression(np.array(detecciones),Ni,Nj, 0.3)

# Visualizamos las detecciones
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

for i, j in detecciones:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/a2f64968f9c2b798e70df941e450f34800cf8170.png)
:::
:::

::: {.cell .markdown}
### 5.2 Test con 3 rostros {#52-test-con-3-rostros}
:::

::: {.cell .code execution_count="60"}
``` python
test_image = plt.imread('./imagenes/3_Rostros.jpg')
test_image = color.rgb2gray(test_image)
test_image = rescale(test_image,0.5)
test_image.shape
```

::: {.output .execute_result execution_count="60"}
    (292, 443)
:::
:::

::: {.cell .code execution_count="61"}
``` python
# Visualizamos la imagen
# Buscamos la escala de los rostros
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')

scale = 2.47
Ni, Nj = positive_patche_shape

ax.add_patch(plt.Rectangle((0, 0), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
plt.show()
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/680950f50a291bca0b73ab217457085d55ec1806.png)
:::
:::

::: {.cell .code execution_count="62"}
``` python
# Utiliza la función de ventana deslizante en una imagen de prueba.
# zip(*...) toma las tuplas generadas y las descompone en índices y parches.
indices, patches = zip(*detector.sliding_window(test_image, positive_patche_shape,scale=scale))

# Calcula las características HOG para cada parche y las almacena en un array.
patches_hog = np.array([feature.hog(patch) for patch in patches])

# Muestra la forma del array de características HOG.
patches_hog.shape
```

::: {.output .execute_result execution_count="62"}
    (10150, 1215)
:::
:::

::: {.cell .code execution_count="63"}
``` python
# Predicción
labels = (model_saved.predict_proba(patches_hog)[:,1]>=thresholdAde).astype(int)
labels.sum()
```

::: {.output .execute_result execution_count="63"}
    158
:::
:::

::: {.cell .code execution_count="64"}
``` python
Ni, Nj = (int(scale*s) for s in positive_patche_shape)
indices = np.array(indices)
detecciones = indices[labels == 1]
detecciones = detector.non_max_suppression(np.array(detecciones),Ni,Nj, 0.3)

# Visualizamos las detecciones
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

for i, j in detecciones:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/d60ae345e423b5a9ab6ac7088c68055eb8541ebd.png)
:::
:::

::: {.cell .markdown}
### 5.3 Test con muchos rostro {#53-test-con-muchos-rostro}
:::

::: {.cell .code execution_count="65"}
``` python
test_image = plt.imread('./imagenes/Central.jpg')
test_image = color.rgb2gray(test_image)
test_image = rescale(test_image,0.5)
test_image.shape
```

::: {.output .execute_result execution_count="65"}
    (345, 512)
:::
:::

::: {.cell .code execution_count="66"}
``` python
# Visualizamos la imagen
# Buscamos la escala de los rostros
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')

scale = 0.57
Ni, Nj = (int(scale * s) for s in positive_patche_shape)

ax.add_patch(plt.Rectangle((0, 0), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
plt.show()
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/ca392a944007597b8ccc961128eb95bd3d30eec2.png)
:::
:::

::: {.cell .code execution_count="67"}
``` python
# Utiliza la función de ventana deslizante en una imagen de prueba.
# zip(*...) toma las tuplas generadas y las descompone en índices y parches.
indices, patches = zip(*detector.sliding_window(test_image, positive_patche_shape ,scale=scale))

# Calcula las características HOG para cada parche y las almacena en un array.
patches_hog = np.array([feature.hog(patch) for patch in patches])

# Muestra la forma del array de características HOG.
patches_hog.shape
```

::: {.output .execute_result execution_count="67"}
    (37045, 1215)
:::
:::

::: {.cell .code execution_count="68"}
``` python
# Predicción
labels = (model_saved.predict_proba(patches_hog)[:,1]>=thresholdAde).astype(int)
labels.sum()
```

::: {.output .execute_result execution_count="68"}
    86
:::
:::

::: {.cell .code execution_count="69"}
``` python
Ni, Nj = (int(scale*s) for s in positive_patche_shape)
indices = np.array(indices)
detecciones = indices[labels == 1]
detecciones = detector.non_max_suppression(np.array(detecciones),Ni,Nj, 0.3)

# Visualizamos las detecciones
fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

for i, j in detecciones:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
```

::: {.output .display_data}
![](vertopal_2ea61ea6364f4edebd5db3a0ba99f1bd/024f662b456c56527418bd38b2fccccb37bfe27d.png)
:::
:::
