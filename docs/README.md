# Guia Turistica de Colombia — RAG con LangChain

Aplicacion de pregunta-respuesta sobre turismo colombiano construida con el patron **RAG (Retrieval-Augmented Generation)**. Combina una base de datos vectorial con un modelo de lenguaje de Google Gemini para responder preguntas en espanol usando articulos de Wikipedia como fuente de conocimiento.

---

## Indice

1. [Arquitectura general](#1-arquitectura-general)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Conceptos clave](#3-conceptos-clave)
   - [Que es RAG](#que-es-rag)
   - [Runnables](#runnables)
   - [Retrievers](#retrievers)
   - [Vector Store](#vector-store)
   - [Embeddings](#embeddings)
   - [LCEL (LangChain Expression Language)](#lcel-langchain-expression-language)
4. [Documentacion de codigo](#4-documentacion-de-codigo)
   - [rag_chain.py](#rag_chainpy)
   - [app.py](#apppy)
5. [Configuracion y dependencias](#5-configuracion-y-dependencias)
6. [Como ejecutar la aplicacion](#6-como-ejecutar-la-aplicacion)
7. [Preguntas de ejemplo para la aplicacion](#7-preguntas-de-ejemplo-para-la-aplicacion)

---

## 1. Arquitectura general

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI  (app.py)                        │
│   Entrada del usuario → Historial de chat → Respuesta           │
│   Sidebar con destinos disponibles                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │  chain.invoke(pregunta)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PIPELINE RAG  (rag_chain.py)                    │
│                                                                   │
│   Pregunta del usuario                                            │
│        │                                                          │
│        ▼                                                          │
│   RunnableParallel                                                │
│   ├── rama "context":                                             │
│   │      Retriever  ──►  format_docs  ──►  texto con contexto    │
│   └── rama "question":                                            │
│              RunnablePassthrough  ──►  pregunta original          │
│        │                                                          │
│        ▼                                                          │
│   ChatPromptTemplate  (prompt con contexto + pregunta)            │
│        │                                                          │
│        ▼                                                          │
│   ChatGoogleGenerativeAI  (gemini-2.5-flash)                     │
│        │                                                          │
│        ▼                                                          │
│   StrOutputParser  ──►  respuesta en texto plano                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │  busqueda vectorial
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CHROMA VECTOR DB  (chroma_db/)                  │
│   Fragmentos de Wikipedia con embeddings gemini-embedding-001    │
│   Fuentes: Medellin, Cartagena, Bogota, Santa Marta,             │
│            Tayrona, Zona Cafetera, Ciudad Perdida                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Estructura del proyecto

```
rag_turismo/
├── app.py              # Interfaz web con Streamlit
├── rag_chain.py        # Logica RAG: carga de datos, pipeline, retriever
├── requirements.txt    # Dependencias de Python
├── .env                # Variables de entorno (GOOGLE_API_KEY)
├── chroma_db/          # Base de datos vectorial persistida
│   ├── chroma.sqlite3  # Archivo SQLite con vectores
│   └── <uuid>/         # Coleccion de embeddings
└── docs/
    └── README.md       # Esta documentacion
```

---

## 3. Conceptos clave

### Que es RAG

**RAG (Retrieval-Augmented Generation)** es un patron arquitectonico que mejora las respuestas de un LLM al enriquecerlas con informacion externa relevante antes de generar la respuesta. El flujo es:

```
Pregunta → Buscar documentos relevantes → Inyectar en el prompt → LLM genera respuesta
```

Sin RAG, el LLM solo usaria su conocimiento pre-entrenado. Con RAG, el modelo recibe contexto actualizado y especifico del dominio, lo que reduce alucinaciones y permite responder con datos reales.

---

### Runnables

Los **Runnables** son los bloques fundamentales del pipeline en LangChain. Cada runnable expone el metodo `.invoke()` y puede encadenarse con otros usando el operador `|`. La cadena completa de esta aplicacion tiene **7 runnables**:

| # | Tipo | Rol en el pipeline |
|---|------|--------------------|
| 1 | `VectorStoreRetriever` | Busca los 3 fragmentos de Wikipedia mas similares a la pregunta usando similitud coseno |
| 2 | `RunnableLambda(format_docs)` | Convierte la lista de objetos `Document` en un string de texto plano para el prompt |
| 3 | `RunnablePassthrough()` | Pasa la pregunta original sin ninguna modificacion al siguiente paso |
| 4 | `RunnableParallel(context, question)` | Ejecuta las ramas de contexto (R1+R2) y pregunta (R3) en paralelo y agrupa los resultados en un diccionario |
| 5 | `ChatPromptTemplate` | Construye el prompt final interpolando `{context}` y `{question}` en la plantilla de guia turistica |
| 6 | `ChatGoogleGenerativeAI` | Envia el prompt a Gemini 2.5 Flash y obtiene la respuesta del LLM |
| 7 | `StrOutputParser` | Extrae el texto plano del objeto de respuesta del LLM |

La cadena LCEL en codigo:

```python
# rag_chain.py — lineas 118-126
chain = (
    RunnableParallel(                          # Runnable 4
        context=retriever | format_runnable,   # Runnables 1 + 2
        question=RunnablePassthrough(),         # Runnable 3
    )
    | prompt                                   # Runnable 5
    | llm                                      # Runnable 6
    | output_parser                            # Runnable 7
)
```

---

### Retrievers

Un **Retriever** es el componente responsable de recuperar documentos relevantes dado un texto de entrada. La aplicacion soporta cuatro estrategias configurables a traves del parametro `retriever_type` de `build_retriever()`:

---

#### 1. Similarity (por defecto)

```python
vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

Busca los fragmentos mas cercanos a la pregunta usando **similitud coseno** entre embeddings. Rapido y directo. Devuelve los `k=3` fragmentos con mayor similitud semantica. Es el punto de partida mas simple y el que se usa si no se especifica otro tipo.

---

#### 2. MMR — Maximum Marginal Relevance

```python
vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.6},
)
```

Variante de busqueda vectorial que **penaliza la redundancia** entre los fragmentos devueltos. Primero recupera `fetch_k=10` candidatos por similitud y luego selecciona iterativamente los `k=4` que maximizan la relevancia y minimizan la similitud entre ellos. El parametro `lambda_mult` (0 = maxima diversidad, 1 = maxima relevancia) controla el balance. Util cuando varios fragmentos de Wikipedia tratan el mismo subtema.

---

#### 3. MultiQueryRetriever

```python
MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
```

Usa el **LLM para generar automaticamente 3 reformulaciones** de la pregunta original. Lanza las 3 busquedas en paralelo sobre el retriever base y une los resultados (eliminando duplicados). Aumenta el recall cuando la pregunta del usuario es ambigua o podria formularse de distintas formas.

---

#### 4. ContextualCompressionRetriever

```python
compressor = LLMChainExtractor.from_llm(llm)
ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
```

Recupera fragmentos con el retriever base y luego **usa el LLM para extraer solo las oraciones relevantes** de cada fragmento (compresion). Reduce el ruido en el contexto enviado al modelo final: en vez de pasar parrafos enteros, el compressor filtra y retiene solo las partes directamente utiles para la pregunta.

---

**Resumen comparativo:**

| Tipo | Ventaja principal | Costo extra |
|------|-------------------|-------------|
| `similarity` | Rapidez, sin llamadas adicionales al LLM | Ninguno |
| `mmr` | Diversidad de fragmentos | Ninguno (solo vectores) |
| `multi_query` | Mayor recall con preguntas ambiguas | 1 llamada al LLM para generar variantes |
| `compression` | Contexto mas limpio y preciso | 1 llamada al LLM por fragmento recuperado |

El retriever se usa dos veces en `app.py`:
1. Dentro del pipeline `chain.invoke(question)` para generar la respuesta.
2. Directamente via `retriever.invoke(question)` para mostrar al usuario los fragmentos fuente usados como contexto.

---

### Vector Store

La **base de datos vectorial** (Chroma) almacena cada fragmento de texto junto con su vector de embeddings. Permite busqueda semantica: en vez de buscar palabras exactas, busca por significado.

- **Motor:** ChromaDB con persistencia en SQLite (`chroma_db/chroma.sqlite3`)
- **Contenido:** fragmentos de 400 caracteres con solapamiento de 50 caracteres extraidos de 7 articulos de Wikipedia en espanol
- **Proceso de carga:** ejecutado una sola vez; si `chroma.sqlite3` ya existe, se omite la recarga

Destinos indexados:

| Articulo de Wikipedia |
|-----------------------|
| Medellin |
| Cartagena de Indias |
| Bogota |
| Santa Marta Colombia |
| Parque Nacional Natural Tayrona |
| Zona cafetera de Colombia |
| Ciudad Perdida Colombia |

---

### Embeddings

Los **embeddings** son representaciones numericas (vectores de alta dimension) de fragmentos de texto que capturan su significado semantico. Dos textos con significados similares tendran vectores con angulo pequeno (alta similitud coseno).

- **Modelo usado:** `gemini-embedding-001` de Google
- **Usados en dos momentos:**
  1. Al crear la base vectorial: para convertir cada fragmento de Wikipedia en un vector y almacenarlo
  2. En cada consulta: para convertir la pregunta del usuario en un vector y buscarlo en la base

---

### LCEL (LangChain Expression Language)

LCEL es la sintaxis declarativa de LangChain para componer pipelines. Usa el operador `|` (pipe) para encadenar runnables, de forma similar a unix pipes. Ventajas:

- Ejecucion paralela automatica con `RunnableParallel`
- Soporte nativo para streaming
- Trazabilidad con LangSmith
- Legibilidad: el codigo expresa claramente el flujo de datos

---

## 4. Documentacion de codigo

### `rag_chain.py`

**Proposito:** contiene toda la logica del backend RAG — carga de datos, construccion del pipeline y exposicion del retriever.

---

#### Constante `TEMAS_WIKIPEDIA` (linea 15)

```python
TEMAS_WIKIPEDIA = [
    "Medellin",
    "Cartagena de Indias",
    ...
]
```

Lista de consultas que se usaran para descargar articulos de Wikipedia en espanol. Cada entrada se pasa a `WikipediaLoader` con `lang="es"`.

---

#### Constante `DB_PATH` (linea 25)

```python
DB_PATH = "./chroma_db"
```

Ruta relativa donde ChromaDB persiste la base vectorial. Si se cambia la ubicacion de ejecucion del script, esta ruta puede fallar; se recomienda usar `os.path.dirname(__file__)` para paths absolutos en produccion.

---

#### Funcion `create_vector_db()` (lineas 28-67)

```python
def create_vector_db():
    """Descarga los articulos de Wikipedia y construye la base vectorial."""
```

**Flujo:**
1. Verifica si `chroma_db/chroma.sqlite3` ya existe. Si existe, retorna inmediatamente (idempotente).
2. Carga un articulo por cada tema en `TEMAS_WIKIPEDIA` usando `WikipediaLoader` con `load_max_docs=1`.
3. Divide cada articulo con `RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)`.
4. Genera embeddings con `GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")`.
5. Inserta los fragmentos en Chroma en lotes de 80 para respetar el limite de 100 req/min de la API gratuita de Google, esperando 65 segundos entre lotes.

**Por que lotes de 80 y 65 segundos?** La API gratuita de `gemini-embedding-001` tiene un limite de 100 solicitudes por minuto. Procesar en grupos de 80 con pausa de 65s garantiza no superar ese limite.

---

#### Funcion `build_retriever()` (lineas 75-114)

```python
def build_retriever(vectorstore, llm, retriever_type: str = "similarity"):
    """Construye el retriever segun el tipo solicitado."""
```

Fabrica el retriever adecuado segun `retriever_type`. Acepta `"similarity"`, `"mmr"`, `"multi_query"` o `"compression"`. Todos parten del mismo `base_retriever` (similarity, k=4) excepto `"mmr"`, que reconfigura directamente el vector store. Ver la seccion [Retrievers](#retrievers) para la descripcion detallada de cada uno.

---

#### Funcion `build_chain()` (lineas 117-175)

```python
def build_chain(retriever_type: str = "similarity"):
    """Pipeline RAG con LCEL — 7 runnables."""
```

**Flujo:**
1. Llama a `create_vector_db()` para asegurar que la base existe.
2. Instancia `GoogleGenerativeAIEmbeddings` y carga `Chroma` desde el directorio persistido.
3. Configura el LLM `ChatGoogleGenerativeAI` con modelo `gemini-2.5-flash` y `temperature=0.3` (respuestas mas deterministas).
4. Llama a `build_retriever(vectorstore, llm, retriever_type)` para obtener el retriever configurado.
5. Define `format_docs(docs)` como funcion auxiliar que une el contenido de los documentos.
6. Construye el prompt con la plantilla de guia turistica.
7. Encadena todo con LCEL y retorna `(chain, retriever)`.

**Retorno:** tupla `(chain, retriever)` — el chain completo para invocar y el retriever por separado para mostrar fuentes en la UI.

---

#### Funcion auxiliar `format_docs(docs)` (lineas 95-98)

```python
def format_docs(docs):
    if not docs:
        return "No se encontro informacion relevante."
    return "\n\n".join(doc.page_content for doc in docs)
```

Convierte una lista de objetos `Document` (cada uno con `.page_content` y `.metadata`) en un unico string de texto. El resultado es lo que se inyecta en `{context}` dentro del prompt.

---

#### Prompt de la cadena (lineas 103-113)

```
Eres un guia turistico experto en Colombia.
Usa solo la informacion proporcionada. Si no tienes suficiente, dilo.

Informacion:
{context}

Pregunta: {question}

Responde en espanol.
```

La instruccion "usa solo la informacion proporcionada" es clave para evitar alucinaciones: el modelo no debe inventar datos fuera del contexto recuperado.

---

### `app.py`

**Proposito:** interfaz web construida con Streamlit. Gestiona la sesion del usuario, el historial de mensajes y la visualizacion de resultados.

---

#### Configuracion de pagina (lineas 4-8)

```python
st.set_page_config(page_title="Guia Turistica Colombia", page_icon="...", layout="wide")
```

Configura titulo y layout de la pagina de Streamlit.

---

#### Sidebar (lineas 14-20)

Muestra dinamicamente los destinos disponibles iterando sobre `TEMAS_WIKIPEDIA` importado desde `rag_chain`. Informa al usuario la fuente de datos (Wikipedia) y el tipo de retriever.

---

#### Inicializacion del pipeline en sesion (lineas 23-26)

```python
if "chain" not in st.session_state:
    with st.spinner("Cargando informacion turistica desde Wikipedia..."):
        st.session_state.chain, st.session_state.retriever = build_chain()
```

El pipeline se construye **una sola vez por sesion** de Streamlit. Las recargas de pagina no vuelven a construir la cadena gracias a `st.session_state`. Esto es importante porque `build_chain()` puede tardar si la base vectorial aun no existe.

---

#### Historial de mensajes (lineas 28-35)

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Lista de dicts `{"role": "user"/"assistant", "content": "..."}` que mantiene el historial de la conversacion. Se renderiza completo en cada recarga para dar la apariencia de chat continuo.

---

#### Flujo de pregunta-respuesta (lineas 38-61)

Cuando el usuario escribe una pregunta:

1. Se agrega a `st.session_state.messages`.
2. Se invoca `chain.invoke(question)` para obtener la respuesta generada.
3. Se invoca `retriever.invoke(question)` para obtener los fragmentos fuente.
4. Se muestra la respuesta y, en un expander colapsado, los fragmentos de Wikipedia con su titulo y contenido.
5. La respuesta se agrega al historial.

---

## 5. Configuracion y dependencias

### Variables de entorno (`.env`)

```
GOOGLE_API_KEY=<tu_clave_de_google>
```

Necesaria para autenticar con la API de Google Gemini (tanto embeddings como el LLM).

### Dependencias (`requirements.txt`)

| Paquete | Version | Uso |
|---------|---------|-----|
| `langchain` | 1.0.0 | Framework de orquestacion RAG |
| `langchain-google-genai` | - | Integracion con Gemini (LLM + embeddings) |
| `langchain-community` | - | WikipediaLoader, Chroma |
| `langchain-text-splitters` | - | RecursiveCharacterTextSplitter |
| `python-dotenv` | - | Carga de `.env` |
| `streamlit` | - | Interfaz web |
| `chromadb` | - | Base de datos vectorial |
| `wikipedia` | - | Cliente de la API de Wikipedia |

### Modelos de Google

| Modelo | Uso |
|--------|-----|
| `gemini-embedding-001` | Generacion de embeddings (indexacion y busqueda) |
| `gemini-2.5-flash` | Generacion de respuestas (LLM) |

---

## 6. Como ejecutar la aplicacion

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Crear el archivo .env con la API key de Google
echo "GOOGLE_API_KEY=tu_clave_aqui" > .env

# 3. Ejecutar la aplicacion
streamlit run app.py
```

En el primer arranque, la aplicacion descargara los articulos de Wikipedia y construira la base vectorial. Esto puede tomar varios minutos por los limites de la API gratuita. En arranques posteriores la base se carga directamente desde `chroma_db/`.

---

## 7. Preguntas de ejemplo para la aplicacion

Las siguientes preguntas estan disenadas para aprovechar la informacion indexada en la base vectorial. Estan organizadas por destino y por tipo de consulta.

### Medellin

- Que hacer en Medellin en un fin de semana?
- Por que Medellin fue elegida ciudad mas innovadora del mundo?
- Cuales son los barrios mas turisticos de Medellin?
- Como funciona el sistema de transporte de Medellin?
- Que es el metrocable de Medellin y por que es importante?

### Cartagena de Indias

- Que hay que ver en el centro historico de Cartagena?
- Por que Cartagena fue declarada Patrimonio de la Humanidad por la UNESCO?
- Cuales son las playas mas cercanas a Cartagena?
- Que son las Murallas de Cartagena y cual es su historia?
- Cual es la mejor epoca del ano para visitar Cartagena?

### Bogota

- Que museos se pueden visitar en Bogota?
- Que es la Candelaria en Bogota?
- Como es el clima en Bogota y como debo prepararme?
- Que significa Bogota en lengua indigena?
- Cuales son los principales parques o espacios verdes de Bogota?

### Santa Marta

- Que es lo mas destacado de Santa Marta?
- Santa Marta es la ciudad mas antigua de Colombia?
- Como llegar desde Santa Marta al Parque Tayrona?
- Que playas tiene Santa Marta?

### Parque Nacional Natural Tayrona

- Como se llega al Parque Tayrona?
- Que animales se pueden ver en el Parque Tayrona?
- Se puede acampar en el Parque Tayrona?
- Por que el Parque Tayrona es importante para la cultura indigena?
- Cuales son los senderos disponibles dentro del parque?

### Zona Cafetera

- Que departamentos forman la zona cafetera de Colombia?
- Por que el cafe colombiano es tan reconocido internacionalmente?
- Que se puede hacer en el Eje Cafetero ademas de tours de cafe?
- Que es el Paisaje Cultural Cafetero y por que fue declarado Patrimonio de la Humanidad?

### Ciudad Perdida

- Como se llega a Ciudad Perdida?
- Cuantos dias dura el trekking a Ciudad Perdida?
- Que cultura construyo Ciudad Perdida?
- Cual es la diferencia entre Ciudad Perdida y Machu Picchu?
- Quien descubrio Ciudad Perdida en tiempos modernos?

### Preguntas comparativas y generales

- Cual es el destino mas recomendado para una primera visita a Colombia?
- Que destinos de Colombia son Patrimonio de la Humanidad?
- En que destino colombiano puedo encontrar mejor biodiversidad?
- Cuales son los destinos ideales para turismo de aventura en Colombia?
- Que destinos coloniales tiene Colombia para visitar?
- Donde puedo combinar playa y naturaleza en Colombia?
- Que sabe sobre la historia prehispanica de Colombia?
