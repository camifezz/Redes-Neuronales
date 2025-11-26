# Redes-Neuronales - TP3
![](logo-fiuba.png)

|          |                   |
|---------------|------------------------|
| Nombre        | Camila Fernández Marchitelli     |
| Padrón          | 102515               |
| Año       | 2025 |
| Mail          | cfernandezm@fi.uba.ar    |

## Tabla de Contenidos
- [Redes de Kohonen](#red-de-kohonen)
- [Ejercicios](#ejercicios)


## Redes de Kohonen

Las redes de Kohonen, también conocidas como mapas auto-organizados, son un tipo de red neuronal artificial no supervisada que permite hacer agrupación y visualización de datos. Reciben vectores numéricos de entrada y los organizan, agrupan y proyectan sobre un mapa 1D o 2D.

La idea central es que la red aprende a construir un mapa de características donde:
- Patrones de entrada similares activan neuronas vecinas en el mapa.
- Patrones diferentes terminan representados en zonas más alejadas.

De esta forma, la red intenta preservar la topología del espacio de datos original: la relación de “cercanía” entre ejemplos se refleja en la geometría del mapa.

El entrenamiento es no supervisado y se basa en el aprendizaje competitivo:
- Para cada vector de entrada, se selecciona la neurona ganadora (la neurona donde su vector de pesos es más parecido a la entrada).
- Después, se actualizan tanto esa neurona como sus vecinas en la red, moviendo sus pesos hacia el patrón presentado.
- Con el tiempo, las neuronas se especializan en representar distintos “tipos” de patrones, formando regiones o clusters sobre el mapa.

### Componentes de estas redes

- Se  tiene un vector (1D) o matriz (2D) de unidades que reciben info de eventos que  se pueden ordenar (por ejemplo, por posición, valor, etc.).
- Cada unidad calcula una “respuesta” a cada evento.
- Para cada evento, mirás qué unidad responde más fuerte → esa unidad es la imagen de ese evento en el mapa.

### Entradas en la red
Cada entrada a la red es un vector de características:

x = (x₁, x₂, …, xₙ) ∈ ℝⁿ

Ese vector puede representar cualquier cosa:

- un cliente (facturación, frecuencia, etc.),
- un píxel/imagen pequeña,
- un patrón de señal,
- o un ítem abstracto, mientras se pueda escribir como vector numérico.

Las entradas (los vectores o matrices) suelen estar **normalizadas** (por ejemplo, a norma 1) para que la comparación sea más estable.

Cada neurona de la red también tiene un **vector de pesos** (o prototipo) del mismo tamaño:

**m**ᵢ = (*m*ᵢ₁, *m*ᵢ₂, …, *m*ᵢₙ)

### Comparaciones: cómo se elige la neurona “más parecida”
Para cada entrada x:

1. Cada neurona *i* calcula una **medida de similitud** con su vector de pesos **m**ᵢ
    
    Las formas más comunes son:
    
    - **Producto interno** (función discriminante de Kohonen):
        
        $$
        \varphi_i = m_i^T x
        $$
        
        Si **x** y **m**ᵢ están normalizados, esto equivale al **coseno del ángulo** entre ambos (más grande → más parecidos).
        
    - O, de forma equivalente, **distancia euclídea**:
        
        $$
        d_i = \|x - m_i\|
        $$
        
        (más chico → más parecido).


2. Se elige la **neurona ganadora**:
    - Por producto interno:
        
        $$
        k = \arg\max_i \,\varphi_i
        $$
        
    - O por distancia:
        
        $$
        k = \arg\min_i \, d_i
        $$

3. Esa neurona ganadora y sus vecinas en la grilla ajustan sus pesos **acercándose a X**.

Al principio, los pesos de las neuronas se inicializan al azar porque no hay conocimientos.

Algunas de las formas para inicializar los pesos de las neuronas son:

1. **Aleatorio puro**
    - Elegir cada componente de **m**ᵢ como un número aleatorio pequeño (por ejemplo, uniforme entre 0 y 1)
    - y después normalizarlos.
2. **Aleatorio a partir de los datos** (un poco más prolijo)
    - Elegir **m**ᵢ como vectores tomados al azar del propio conjunto de datos de entrada.
    - Esto hace que desde el principio estén “dentro” de la nube de datos.

### ¿Qué pasa con los pesos?
En esta parte es donde entra el aprendizaje.
Para cada entrada $x(t)$:

1. Se busca la **neurona ganadora**:
    
    $k = \arg\min_i \|x(t) - m_i(t)\|$
    
    (o equivalente con producto interno).
    
2. Se actualizan los pesos de la ganadora **y de sus vecinas**:
    
    $m_i(t+1) = m_i(t) + \alpha(t)\, h_{ki}(t)\, (x(t) - m_i(t))$
    
    (o una versión normalizada, como en la ecuación (3) del paper).
    
- $\alpha(t)$: tasa de aprendizaje (va bajando con el tiempo).
- $h_{ki}(t)$: cuánto influye el ganador $k$ sobre la neurona $i$ (grande si es vecina, chico si está lejos).

Con cada iteración:

- los **pesos aleatorios del inicio** se van moviendo,
- cada neurona se especializa en una “zona” del espacio de datos,
- y la grilla completa se **autoorganiza** formando el mapa.


La forma final del mapa de Kohonen depende de la topología de la rejilla de neuronas (1D, 2D, vecinos, bordes), de la estructura de los datos de entrada (su distribución y geometría), de los parámetros de entrenamiento (radio de vecindad y tasa de aprendizaje) y de la inicialización de los pesos. El mapa se deforma para ajustarse a la “forma” del espacio de datos, respetando lo mejor posible las relaciones de vecindad.



## Ejercicios

### 1. Construya una red de Kohonen de 2 entradas que aprenda una distribución uniforme dentro del círculo unitario. Mostrar el mapa de preservación de topología. Probar con distribuciones uniformes dentro de otras figuras geométricas.



### 2.  Resuelva (aproximadamente) el “Traveling salesman problem” para 200 ciudades con una red de Kohonen.

En principio necesito tantas neuronas como ciudades, mas neuronas tengo mas W

### En el campus encontrará el archivo “datos_para_clustering.mat” que contiene una matriz de datos de 500 mediciones de una variable de 100 dimensiones.
a) Utilice una red de Kohonen para reducir la dimensionalidad de los datos.

b) Verifique la presencia de clusters, e indique cuantos puede visualizar, haciendo uso
de la matriz U.

### COmentarios adicionales
Para el ejercicio 3, podemos usar todo el codigo del ejercicio 1 y el entrenamiento y demas es lo mismo.
Hay que probar varias combinaciones para ver la cantidad de clusters que hay (puedo graficarlo con un heatmap)

rveiga@fi.uba.ar -> ver si hay algun paper interesante y mandarselo por mail.