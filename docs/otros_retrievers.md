# Otros Retrievers en LangChain

Retrievers que no se usaron en el proyecto pero que son relevantes para sistemas RAG mas avanzados.

---

## 1. BM25Retriever

**Que es:** retriever basado en el algoritmo clasico de recuperacion de informacion **BM25** (Best Match 25). No usa embeddings ni vectores — trabaja con frecuencia de terminos (TF-IDF extendido) y longitud del documento.

**Como funciona:** dado un texto de consulta, calcula un puntaje para cada documento basandose en cuantas veces aparecen los terminos de la consulta en el documento, penalizando documentos muy largos. Es una busqueda lexica, no semantica.

**Cuando usarlo:**
- Cuando los terminos exactos importan (nombres propios, codigos, siglas).
- Como componente de un `EnsembleRetriever` junto a un retriever vectorial.
- Cuando no se tiene acceso a un modelo de embeddings o se quiere bajo costo computacional.

**Ejemplo:**
```python
from langchain_community.retrievers import BM25Retriever

retriever = BM25Retriever.from_documents(docs, k=4)
result = retriever.invoke("Parque Tayrona fauna")
```

---

## 2. EnsembleRetriever

**Que es:** retriever que **combina los resultados de multiples retrievers** usando el algoritmo **Reciprocal Rank Fusion (RRF)**. Fusiona las listas de documentos ponderando cada resultado por la posicion que ocupa en cada lista.

**Como funciona:** ejecuta varios retrievers en paralelo, obtiene una lista ordenada de cada uno y las fusiona. Un documento que aparece en el top-3 de varios retrievers sube en el ranking final. Los pesos (`weights`) permiten dar mas importancia a un retriever que a otro.

**Cuando usarlo:**
- Para combinar busqueda lexica (BM25) con busqueda semantica (vectorial) y obtener lo mejor de ambos mundos.
- Cuando distintas consultas se benefician de distintas estrategias de recuperacion.

**Ejemplo:**
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=4)
vector = vectorstore.as_retriever(search_kwargs={"k": 4})

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.4, 0.6],
)
result = ensemble.invoke("historia de Cartagena")
```

---

## 3. SelfQueryRetriever

**Que es:** retriever que usa el **LLM para traducir la pregunta en lenguaje natural a una consulta estructurada** con filtros sobre los metadatos del vector store. Combina busqueda semantica con filtrado exacto por metadatos.

**Como funciona:** se le proporciona al LLM una descripcion de los metadatos disponibles (ej. `fuente`, `fecha`, `tema`). El LLM analiza la pregunta del usuario y genera automaticamente: (1) el texto de busqueda semantica y (2) los filtros de metadatos a aplicar. El resultado es una consulta hibrida precisa.

**Cuando usarlo:**
- Cuando los documentos tienen metadatos ricos (fecha, autor, categoria, idioma).
- Para preguntas como "muestra articulos de Medellin publicados despues de 2020".
- Cuando el usuario filtra por atributos especificos sin saberlo.

**Ejemplo:**
```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_fields = [
    AttributeInfo(name="source", description="Fuente del documento", type="string"),
    AttributeInfo(name="title", description="Titulo del articulo", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Articulos de turismo colombiano",
    metadata_field_info=metadata_fields,
)
result = retriever.invoke("articulos sobre Bogota")
```

---

## 4. TimeWeightedVectorStoreRetriever

**Que es:** retriever vectorial que **pondera la similitud semantica con la antiguedad del documento**. Los documentos mas recientes reciben un boost en el ranking final.

**Como funciona:** combina el score de similitud coseno con un factor de decaimiento exponencial basado en cuando fue accedido o insertado el documento por ultima vez. La formula es: `score = semantic_score + (1 - decay_rate) ^ horas_desde_ultimo_acceso`.

**Cuando usarlo:**
- Bases de conocimiento que se actualizan frecuentemente (noticias, logs, tickets).
- Aplicaciones donde la informacion reciente es mas confiable o relevante que la antigua.
- Agentes que necesitan "memoria" con olvido gradual del pasado.

**Ejemplo:**
```python
from langchain.retrievers import TimeWeightedVectorStoreRetriever

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.01,   # 0 = sin decaimiento, 1 = olvido inmediato
    k=4,
)
result = retriever.invoke("eventos recientes en Colombia")
```

---

## 5. ParentDocumentRetriever

**Que es:** retriever que **indexa fragmentos pequenos para la busqueda pero devuelve los documentos padre (mas grandes) como contexto**. Separa la granularidad de indexacion de la granularidad de recuperacion.

**Como funciona:** divide cada documento en dos niveles: fragmentos "hijo" (pequenos, para embedding preciso) y fragmentos "padre" (grandes, para contexto rico). La busqueda vectorial se hace sobre los hijos, pero el retriever devuelve el padre correspondiente. Los padres se almacenan en un `InMemoryStore` o `LocalFileStore`.

**Cuando usarlo:**
- Cuando fragmentos muy pequenos dan embeddings precisos pero pierden el contexto necesario para responder bien.
- Para equilibrar precision en la recuperacion con riqueza del contexto entregado al LLM.
- Documentos tecnicos, legales o academicos donde el parrafo completo es necesario para interpretar una oracion.

**Ejemplo:**
```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
result = retriever.invoke("flora del Parque Tayrona")
```

---

## Comparativa general

| Retriever | Tipo de busqueda | Requiere LLM | Caso de uso tipico |
|-----------|-----------------|--------------|-------------------|
| BM25 | Lexica (TF-IDF) | No | Terminos exactos, bajo costo |
| Ensemble | Hibrida (fusion) | No | Combinar BM25 + vectorial |
| SelfQuery | Semantica + filtros | Si | Metadatos ricos, filtros naturales |
| TimeWeighted | Semantica + tiempo | No | Bases de conocimiento dinamicas |
| ParentDocument | Semantica multi-nivel | No | Contexto amplio, indexacion fina |
