# Recomendador explicable con factorización matricial evolutiva

Código de la tesis sobre un sistema de recomendación explicable basado en filtrado
colaborativo, programación evolutiva y visualización (Universidad Nacional de San
Agustín de Arequipa).

El sistema ajusta una factorización matricial con sesgos mediante un algoritmo
evolutivo híbrido (selección elitista, cruce uniforme, mutación gaussiana y un paso de
mutación dirigida por el residuo), explica cada recomendación con la descomposición
del *score* y con evidencia del historial del usuario, y evalúa esas explicaciones con
métricas *offline*.

## Contenido

| Ruta | Qué contiene |
|---|---|
| `evolutionary_cf.py` | Modelo, carga de datos, evidencia, confianza y visualizaciones |
| `explanation_metrics.py` | Partición, métricas de calidad explicativa y de calidad del recomendador |
| `evolutionary_cf_multi_dataset_colab.ipynb` | Cuaderno de demostración: entrena, recomienda y genera las figuras |
| `analisis/` | Scripts que reproducen las tablas y figuras del Capítulo 5 |
| `datasets/` | Conjuntos de datos (no se distribuyen; ver `datasets/README.md`) |

## Requisitos

Python 3.10 o posterior. El modelo y las métricas usan solo la biblioteca estándar.
Las figuras requieren Matplotlib y los scripts de `analisis/`, NumPy.

```
pip install -r requirements.txt
```

## Uso

Entrenar con la configuración de la tesis y generar las figuras de un usuario:

```
python evolutionary_cf.py --movielens-100k datasets/ml-100k --user 224 --explain-plots --user-explain-plots
python evolutionary_cf.py --dataset bookcrossing --data-dir datasets --user 3331
python evolutionary_cf.py --dataset lastfm --data-dir datasets/hetrec-lastfm --user 333
```

Evaluar las métricas sobre una partición 80/20:

```
python explanation_metrics.py --movielens-100k datasets/ml-100k --evaluate-stability
```

Los valores por defecto son los de la tabla de hiperparámetros de la tesis: 6 factores,
población de 25, 120 generaciones, sigma de mutación 0.08, tasa de mutación 0.02, tasa
de cruce 0.60, tasa de guía 0.01, mini-lote de 8192 y escala de inicialización 0.3.
Con `--guidance-rate 0` se obtiene la variante sin mutación dirigida.

## Reproducción de los resultados

Todos los scripts se ejecutan desde `analisis/`. La partición es 80/20 estratificada
por usuario; la semilla de partición y la del optimizador son independientes.

1. Entrenar y evaluar sobre toda la población de usuarios. Es el paso costoso, de
   varias horas por conjunto en Python puro; se puede interrumpir y reanudar.

   ```
   python corridas_poblacion.py movielens base gamma particiones
   python corridas_poblacion.py bookcrossing base
   python corridas_poblacion.py lastfm base
   python control_trayectoria.py movielens
   python control_trayectoria.py bookcrossing
   python control_trayectoria.py lastfm
   ```

2. Calcular la estabilidad entre semillas y las tablas. Estos pasos leen los resultados
   anteriores y no reentrenan.

   ```
   python estabilidad_poblacion.py
   python tablas_pi1.py
   python tablas_pi2.py
   python tablas_pi3.py
   python tablas_pi4.py
   python tablas_pi5.py
   python control_trayectoria.py resumen
   ```

3. Controles y figuras.

   ```
   python diagnostico_convergencia.py
   python controles_pi4.py
   python sesgos_propios.py
   python barrido_sgd_poblacion.py
   python casos_estudio.py movielens
   python casos_estudio.py bookcrossing
   python casos_estudio.py lastfm
   ```

| Script | Resultado de la tesis |
|---|---|
| `corridas_poblacion.py` | Modelos y métricas por usuario: cinco semillas del modelo adoptado, alternativas de la ablación, descenso de gradiente, solo sesgos, barrido de gamma y cinco particiones de MovieLens |
| `tablas_pi1.py` | Ablación del operador de mutación en los tres conjuntos |
| `tablas_pi2.py` | Usuarios evaluados, selección del caso típico y proporciones sobre la población |
| `casos_estudio.py` | Figuras de los casos de estudio |
| `tablas_pi3.py` | Calidad del recomendador, métricas explicativas, necesidad y suficiencia, fidelidad de la evidencia y sensibilidad de la confianza |
| `estabilidad_poblacion.py`, `tablas_pi4.py` | Estabilidad entre semillas, variabilidad entre semillas y particiones, tolerancia a la escasez |
| `diagnostico_convergencia.py` | Convergencia respecto del propio *fitness* |
| `controles_pi4.py`, `sesgos_propios.py` | Controles de la estabilidad |
| `tablas_pi5.py`, `control_trayectoria.py`, `barrido_sgd_poblacion.py` | Comparación con el descenso de gradiente y término de explicabilidad en el *fitness* |

`analisis/modelos/` guarda los parámetros del modelo adoptado
con la semilla 42, que usan los casos de estudio.
