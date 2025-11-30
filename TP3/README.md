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

Para hacer este ejercicio, se usó una red de Kohonen con 144 neuronas, donde cada neurona tiene asociado su vector de pesos.
La tasa de aprendizaje inicial es de 0.5.

Para documentar el comportamiento de la red, se generaron cuatro distribuciones y se guardó el mapa final de neuronas.

**Cuadrado**

![Mapa Kohonen cuadrado](ejercicio-1/figuras/cuadrado.png)

**Círculo**

![Mapa Kohonen círculo](ejercicio-1/figuras/circulo.png)

**Parábola**

![Mapa Kohonen parábola](ejercicio-1/figuras/parabola.png)
 

**Hipérbola**

![Mapa Kohonen hipérbola](ejercicio-1/figuras/hiperbola.png)



### 2.  Resuelva (aproximadamente) el “Traveling salesman problem” para 200 ciudades con una red de Kohonen.

`The Traveling salesman problem` se trata de encontrar el recorrido "menos costoso" pasando por todas las ciudades,volviendo a la ciudad de origen.

Para calcular la longitud total del camino propuesto por la red, se usa la distancia euclídea entre cada ciudad, y también asumo que la unidad de medida es km.

Para probar si la red devuelve un buen resultado, elijo probar con pocas ciudades comparando el camino óptimo calculado con fuerza bruta.

Tener en cuenta que para todos los casos se usó: `cantidad de neuronas = cantidad de ciudades`


### 10 ciudades

La longitud del camino mas óptimo para 10 ciudades calculado por fuerza bruta es : `3.194`

Comparando con el resultado que me devuelve la red con:
- `Cantidad de iteraciones = 100`
- `La elección inicial de pesos de forma aleatoria `
- `Learning rate = 0.4`

![Mapa 10 ciudades](ejercicio-2/graficos/10-ciudades.png)

![Mapa devuelto por la red](ejercicio-2/graficos/resultado-10-ciudades.png)

### 20 ciudades

Para este caso, no pude calcular el camino más óptimo por fuerza bruta porque tarda mucho tiempo de procesamiento.
Pero se pueden observar algunas cosas con estos parámetros:

- `Cantidad de iteraciones = 200`
- `La elección inicial de pesos de forma aleatoria `
- `Learning rate = 0.4`

![Mapa 20 ciudades](ejercicio-2/graficos/20-ciudades.png)

![resultado 20 ciudades](ejercicio-2/graficos/resultado-20-ciudades.png)

Se arman dos triángulos (cruces) un poco extraños en el lado inferior derecho, puede ser debido a la cantidad de iteraciones o el learning rate.
Entonces, decidí volver a correr el modelo pero ajustado los hiperpaŕametros.

- `Cantidad de iteraciones = 1000`
- `La elección inicial de pesos de forma aleatoria `
- `Learning rate = 0.6`

![resultado 20 ciudades optimizada](ejercicio-2/graficos/resultado-20-ciudades-v2.png)

Se puede observar que se "suaviza" la ruta pero el costo del camino no mejora mucho más.

### 200 ciudades

- `Cantidad de iteraciones = 2000`
- `La elección inicial de pesos de forma aleatoria `
- `Learning rate = 0.6`

![200 ciudades](ejercicio-2/graficos/200-ciudades.png)
![200 ciudades resultado](ejercicio-2/graficos/resultado-200-ciudades.png)

Con los hiperparámetros mencionados arriba, se puede observar una ruta con muchos picos y cruces raros.

Ajustando los hiperparámetros a:

- `Cantidad de iteraciones = 6000`
- `La elección inicial de pesos de forma aleatoria `
- `Learning rate = 0.8`

![200 ciudades resultado 2](ejercicio-2/graficos/resultado-200-ciudades-v2.png)

Con el ajuste de los hiperpárametros se sigue viendo un camino con bastantes picos, y la mejoría de la longitud del camino es mínima, solo mejora un 0.4%, no me parece una mejora significante para haber aumentado un 200% la cantidad de iteraciones y un 33% el learning rate.





### 3. En el campus encontrará el archivo “datos_para_clustering.mat” que contiene una matriz de datos de 500 mediciones de una variable de 100 dimensiones.
a) Utilice una red de Kohonen para reducir la dimensionalidad de los datos.

b) Verifique la presencia de clusters, e indique cuantos puede visualizar, haciendo uso
de la matriz U.

Para este ejercicio se reutilizó mucho del código del ejercicio 1.
Hice varias pruebas con diferentes combinaciones de hiperparámetros para poder identificar la cantidad de clusters.

### Primera ejecución
- `Cantidad de neuronas = 144`
- `Learning rate = 0.5`
- `semilla inicial = 42`
- `Cantidad de iteraciones = 15000`

![Primer resultado](ejercicio-3/graficos/heatmap-1.png)


### Segunda ejecución
- `Cantidad de neuronas = 255`
- `Learning rate = 0.7`
- `semilla inicial = 7`
- `Cantidad de iteraciones = 25000`

![Primer resultado](ejercicio-3/graficos/heatmap-2.png)


### Tercera ejecución
- `Cantidad de neuronas = 324`
- `Learning rate = 0.4`
- `semilla inicial = None`
- `Cantidad de iteraciones = 30000`
![Primer resultado](ejercicio-3/graficos/heatmap-3.png)


Se pueden ver 4 clusters en las 3 ejecuciones variando bastante los hiperparámetros, y ambos en la misma posición (zonas violetas oscuras) delimitadas por las lineas mas claras.

