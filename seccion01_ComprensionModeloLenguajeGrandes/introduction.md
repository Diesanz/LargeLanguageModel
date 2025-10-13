# Large Language Models

# Introdución
Los  modelos  de  lenguaje  grandes  (LLM),  como  los  que  ofrece  ChatGPT  de  OpenAI,  son  modelos  de  
redes  neuronales  profundas  desarrollados  en  los  últimos  años.  Marcaron  el  comienzo  de  una  nueva  era  
para  el  procesamiento  del  lenguaje  natural  (PLN).  Antes  de  la  llegada  de  los  modelos  de  lenguaje  
grandes,  los  métodos  tradicionales  destacaban  en  tareas  de  categorización  como  la  clasificación  de  
correo  no  deseado  y  el  reconocimiento  de  patrones  sencillos,  que  podían  capturarse  con  reglas  
personalizadas  o  modelos  más  simples.  Sin  embargo,  solían  tener  un  rendimiento  inferior  en  tareas  
lingüísticas  que  exigían  capacidades  complejas  de  comprensión  y  generación,  como  analizar  
instrucciones  detalladas,  realizar  análisis  contextuales  o  crear  texto  original  coherente  y  contextualmente  
adecuado. 

Gracias  a  los  avances  en  aprendizaje  profundo,  un  subconjunto  del  aprendizaje  automático  y  la  inteligencia  artificial  (IA)  
centrado  en  redes  neuronales,  los  LLM  se  entrenan  con  grandes  cantidades  de  datos  textuales.  Esto  les  permite  capturar  
información  contextual  más  profunda  y  las  sutilezas  del  lenguaje  humano  en  comparación  con  enfoques  anteriores.  Como  
resultado,  los  LLM  han  mejorado  significativamente  su  rendimiento  en  una  amplia  gama  de  tareas  de  PLN,  como  la  traducción  
de  textos,  el  análisis  de  sentimientos,  la  respuesta  a  preguntas  y  muchas  más

El  éxito  de  los  LLM  se  puede  atribuir  a  la  arquitectura  del  transformador  que
 sustenta  muchos  LLM  y  las  grandes  cantidades  de  datos  con  los  que  se  entrenan  los  LLM,  lo  que  les  permite  capturar  una  
amplia  variedad  de  matices,  contextos  y  patrones  lingüísticos  que  serían  difíciles  de  codificar  manualmente.

## ¿Qué es un LLM?
Un LLM (Large Language Model o modelo de lenguaje extenso) es una red neuronal profunda entrenada con enormes cantidades de texto para comprender y generar lenguaje similar al humano. Se basan en la predicción de la siguiente palabra de una secuencia, lo que les permite captar contexto, estructura y significado.

La palabra “grande” hace referencia tanto al número de parámetros (que pueden llegar a cientos de miles de millones) como a la magnitud de los datos usados en el entrenamiento. Para ello utilizan la arquitectura transformador, que presta atención selectiva a distintas partes del texto y resulta muy eficaz para manejar los matices del lenguaje.

Los LLM forman parte de la inteligencia artificial generativa (GenAI) y representan una aplicación concreta del aprendizaje profundo, que a su vez es un subcampo del aprendizaje automático. Mientras que el aprendizaje automático tradicional requería la extracción manual de características (por ejemplo, en un filtro de spam), el aprendizaje profundo con redes neuronales multicapa permite aprender directamente de los datos sin esa intervención manual.

![Texto alternativo](./imgs/1.1.png)

## Aplicaciones de los LLM
- Qué hacen: los LLM analizan y generan texto no estructurado con gran precisión, por eso se usan en tareas de procesamiento de lenguaje natural.

- Aplicaciones principales: traducción automática, generación de texto creativo y técnico (artículos, ficción, código), análisis de sentimientos y resumen automático de textos.

- Interfaces conversacionales: sirven como núcleo de chatbots y asistentes virtuales avanzados (por ejemplo, ChatGPT o Gemini), y pueden mejorar motores de búsqueda al entender consultas en lenguaje natural.

- Uso en dominios especializados: permiten recuperar conocimientos y resumir grandes volúmenes de documentación en campos como medicina o derecho, y responder preguntas técnicas.

- Valor general: automatizan prácticamente cualquier tarea que implique analizar o generar texto, haciendo la interacción con la tecnología más conversacional, intuitiva y accesible.

## Etapas de construcción y uso de LLM

Desarrollar un LLM desde cero permite comprender su funcionamiento, limitaciones y cómo ajustarlo a tareas o dominios específicos. Los LLM personalizados pueden superar a los modelos de propósito general en aplicaciones concretas y ofrecen ventajas en privacidad, implementación local, reducción de latencia y control sobre el modelo.

El proceso de creación de un LLM se realiza en dos etapas:

1. Preentrenamiento: se entrena el modelo con grandes volúmenes de texto sin etiquetar para que aprenda patrones generales del lenguaje. Esto genera un modelo base o de fundación, como GPT-3, capaz de completar texto y aprender tareas con pocos ejemplos.

2. Ajuste fino: se entrena adicionalmente con datos etiquetados para tareas específicas, como traducción, clasificación o respuesta a instrucciones. Existen dos tipos principales de ajuste fino: de instrucciones (pares de consulta y respuesta) y de clasificación (textos con etiquetas, como spam/no spam).

En resumen, estas etapas permiten construir LLM eficientes y adaptables, desde un modelo base general hasta un modelo especializado para tareas concretas.

![Texto alternativo](./imgs/1.2.png)

## Introducción a la arquitectura del transformador
- La mayoría de los LLM modernos se basan en la arquitectura de transformador, introducida en 2017, diseñada originalmente para traducción automática.

- El transformador tiene dos submódulos principales:

    1. Codificador: procesa el texto de entrada y lo convierte en representaciones vectoriales que capturan la información contextual.

    2. Decodificador: genera el texto de salida a partir de estas representaciones, palabra por palabra.

- Un componente clave es el mecanismo de autoatención, que permite al modelo ponderar la importancia relativa de cada palabra, capturando dependencias de largo alcance y relaciones contextuales.

- Variantes importantes:

    - BERT: utiliza solo el codificador y se especializa en tareas de comprensión y clasificación de texto, prediciendo palabras enmascaradas.

    - GPT: utiliza principalmente el decodificador, diseñado para tareas generativas como traducción, resumen, escritura de código o ficción.

    ![Texto alternativo](./imgs/1.3.png)

- Los modelos GPT destacan por su capacidad de aprendizaje de cero disparos y de pocos disparos, permitiendo realizar tareas desconocidas con pocos o ningún ejemplo.

    ![Texto alternativo](./imgs/1.5.png)

- Relación transformadores vs LLM: todos los LLM actuales basados en texto usan transformadores, pero no todos los transformadores son LLM (también se aplican en visión o tareas no lingüísticas).


![Texto alternativo](./imgs/1.4.png)

## Utilización de grandes conjuntos de datos

- Los LLM populares, como GPT y BERT, se entrenan con conjuntos de datos enormes y diversos, que incluyen miles de millones de palabras en distintos idiomas y temas, tanto textuales como computacionales.

- Por ejemplo, GPT-3 utilizó datos de CommonCrawl, WebText2, Libros1, Libros2 y Wikipedia, sumando más de 500 mil millones de tokens, aunque el entrenamiento real se realizó con 300 mil millones de tokens.

- Un token es una unidad de texto, que puede corresponder aproximadamente a una palabra o signo de puntuación; la conversión de texto a tokens se llama tokenización.

- La escala y diversidad del conjunto de datos permite que los modelos aprendan sintaxis, semántica, contexto y conocimientos generales, aumentando su versatilidad en múltiples tareas.

- El preentrenamiento de estos modelos es costoso y requiere recursos significativos; se estima que GPT-3 costó 4.6 millones de dólares en créditos de computación en la nube.

- Muchos LLM preentrenados están disponibles como modelos de código abierto, que pueden ajustarse para tareas específicas con conjuntos de datos pequeños, reduciendo recursos y mejorando el rendimiento.

## Una mirada cercana a GPT
- GPT significa Generative Pretrained Transformer (Transformador Generativo Preentrenado) y fue presentado por OpenAI en 2018.

- GPT-3 es una versión ampliada con más parámetros y entrenada con un conjunto de datos más amplio; ChatGPT se creó ajustando GPT-3 con un conjunto de datos de instrucciones mediante InstructGPT.

- El preentrenamiento de GPT se basa en la predicción de la siguiente palabra, un enfoque de aprendizaje autosupervisado, que permite usar grandes conjuntos de datos de texto sin etiquetar.

    ![Texto alternativo](./imgs/1.6.png)

- A diferencia del transformador original (codificador + decodificador), GPT utiliza solo el decodificador, funcionando de manera autorregresiva: cada palabra generada depende de las anteriores, mejorando la coherencia del texto.

    ![Texto alternativo](./imgs/1.7.png)

- GPT-3 es un modelo muy grande, con 96 capas de transformador y 175 mil millones de parámetros, frente a las 6 capas del transformador original.

- A pesar de su arquitectura simple, GPT puede realizar tareas complejas como traducción, corrección de texto, clasificación y más, incluso si no se entrenó explícitamente para ellas. Este fenómeno se conoce como comportamiento emergente.

- Las arquitecturas más recientes (como LLaMA de Meta) se basan en los mismos principios de GPT, por lo que comprender GPT sigue siendo relevante.


## Construcción de un modelo de lenguaje grande
El desarrollo del LLM se aborda en tres etapas principales:

1. Implementación de la arquitectura y preprocesamiento de datos, incluyendo la codificación del mecanismo de atención central en todos los LLM.

2. Preentrenamiento del modelo base, donde se genera texto nuevo. Para fines educativos, se usa un conjunto de datos reducido, y también se enseñará a cargar pesos de modelos preentrenados disponibles públicamente.

3. Ajuste fino del modelo base, adaptándolo a tareas específicas como respuesta a consultas o clasificación de texto, transformándolo en un asistente personal o herramienta de análisis.

Se destaca que el preentrenamiento desde cero de un LLM completo es extremadamente costoso, por lo que la implementación práctica en el libro es educativa y simplificada.

Estas etapas permiten comprender todo el ciclo de construcción y adaptación de un LLM, desde los datos crudos hasta un modelo funcional para tareas reales.

![Texto alternativo](./imgs/1.8.png)
