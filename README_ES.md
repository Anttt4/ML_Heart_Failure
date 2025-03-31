# ML_Heart_Failure

Este proyecto de Machine Learning pretende predecir si una persona sería propensa a tener una afección cardiaca teniendo en cuenta una serie de características sobre la salud.

El dataset utilizado es público y proviene de Kaggle (usuario FEDESORIANO -  https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). El dataset tiene 918 filas y 12 columnas. Las columnas son las siguientes:

- ***Age***: Edad del paciente (años)
- ***Sex***: Sexo del paciente (M: Hombre, F: Mujer)
- ***ChestPainType***: Tipo de dolor de pecho (TA: Angina típica, ATA: Angina atípica, NAP: Dolor no anginoso, ASY: Asíntomático)
- ***RestingBP***: Presión arterial en reposo (mm Hg)
- ***Cholesterol***: Colesterol sérico (mm/dl)
- ***FastingBS***: Azúcar en sangre en ayunas (1: si FastingBS > 120 mg/dl, 0: otros)
- ***RestingECG***: Resultado del electrocardiograma en reposo (Normal: Normal, ST: tener anomalía de la onda ST-T (Inversiones de la onda T y/o elevación o depresión del segmento ST of > 0.05 mV), LVH: que muestra hipertrofia ventricular izquierda probable o definitiva según los criterios de Estes)
- ***MaxHR***: Frecuencia cardíaca máxima alcanzada (Valro numérico entre 60 y 202)
- ***ExerciseAngina***:  Angina inducida por el ejercicio (Y: Sí, N: No)
- ***Oldpeak***: Oldpeak = ST (Valor numérico medido en depresión)
- ***ST_Slope***: La pendiente del segmento ST del ejercicio máximo (Up: pendiente ascendente, Flat: plano, Down: pendiente descendente)

El proyecto se desglosa en una primera visualización de los datos para, posteriormente, y mediante una serie de pipelines, realizar el preprocesado, selección de características relevantes, creación y optimización de modelos de Machine Learning. Finalmente se evalúa el modelo optimizado contra el dataset de test.

El repositorio se divide en una carpeta 'main' con este README y una carpeta 'src', que a su vez tiene las siguientes subcarpetas:

- data_sample: dataset empleado
- img: imágenes que se han podido emplear
- models: mejor modelo obtenido
- notebooks: notebooks utilizados
- results_notebook: notebook final
- utils: funciones que se han podido emplear


