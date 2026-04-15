from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda)
from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Temas que se cargarán desde Wikipedia en español
TEMAS_WIKIPEDIA = [
    "Medellín",
    "Cartagena de Indias",
    "Bogotá",
    "Santa Marta Colombia",
    "Parque Nacional Natural Tayrona",
    "Zona cafetera de Colombia",
    "Ciudad Perdida Colombia",
]

DB_PATH = "./chroma_db"


def create_vector_db():
    """Descarga los artículos de Wikipedia y construye la base vectorial."""
    chroma_file = os.path.join(DB_PATH, "chroma.sqlite3")
    if os.path.exists(chroma_file):
        print("Base vectorial ya existe.")
        return

    print("Descargando artículos de Wikipedia...")

    docs = []
    for tema in TEMAS_WIKIPEDIA:
        print(f"  Cargando: {tema}")
        loader = WikipediaLoader(query=tema, lang="es", load_max_docs=1)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Total de fragmentos: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Lotes de 80 para respetar el límite de la API gratuita (100 req/min)
    BATCH_SIZE = 80
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        print(f"  Procesando fragmentos {
              i + 1}–{min(i + BATCH_SIZE, len(chunks))} de {len(chunks)}...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_PATH,
            )
        else:
            vectorstore.add_documents(batch)
        if i + BATCH_SIZE < len(chunks):
            print("  Esperando 65s para respetar el límite de la API...")
            time.sleep(65)

    print("Base vectorial lista.")


def build_retriever(vectorstore, llm, retriever_type: str = "similarity"):
    """
    Construye el retriever según el tipo solicitado.

    Tipos disponibles:
      - "similarity"   : VectorStoreRetriever básico (k=3) — igual al de proyecto_rag
      - "mmr"          : Maximum Marginal Relevance — reduce redundancia entre fragmentos
      - "multi_query"  : MultiQueryRetriever — genera varias reformulaciones de la pregunta
      - "compression"  : ContextualCompressionRetriever — filtra el contenido con el LLM
    """
    base_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    if retriever_type == "mmr":
        # MMR penaliza documentos muy similares entre sí → más diversidad
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.6},
        )

    if retriever_type == "multi_query":
        # El LLM genera 3 variantes de la pregunta y une los resultados
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )

    if retriever_type == "compression":
        # El LLM extrae solo las oraciones relevantes de cada fragmento
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

    # Por defecto: similarity simple (k=3)
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )


def build_chain(retriever_type: str = "similarity"):
    """
    Pipeline RAG con LCEL.

    Runnables:
      1 - Retriever (tipo configurable)       : recupera fragmentos relevantes
      2 - RunnableLambda(format_docs)         : convierte docs a texto plano
      3 - RunnablePassthrough()               : pasa la pregunta sin cambios
      4 - RunnableParallel(...)               : ejecuta context y question en paralelo
      5 - ChatPromptTemplate                  : construye el prompt con contexto + pregunta
      6 - ChatGoogleGenerativeAI              : genera la respuesta
      7 - StrOutputParser                     : extrae el texto de la respuesta
    """
    create_vector_db()

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(embedding_function=embeddings,
                         persist_directory=DB_PATH)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # Runnable 1 — retriever configurable
    retriever = build_retriever(vectorstore, llm, retriever_type)

    # Runnable 2 — convierte la lista de docs a string
    def format_docs(docs):
        if not docs:
            return "No se encontró información relevante."
        return "\n\n".join(doc.page_content for doc in docs)

    format_runnable = RunnableLambda(format_docs)

    # Runnable 5 — prompt de guía turística
    prompt = ChatPromptTemplate.from_template(
        """Eres un guía turístico experto en Colombia.
Usa solo la información proporcionada. Si no tienes suficiente, dilo.

Información:
{context}

Pregunta: {question}

Responde en español."""
    )

    output_parser = StrOutputParser()

    # Cadena LCEL completa
    chain = (
        RunnableParallel(                          # Runnable 4
            context=retriever | format_runnable,   # Runnable 1 + 2
            question=RunnablePassthrough(),         # Runnable 3
        )
        | prompt                                   # Runnable 5
        | llm                                      # Runnable 6
        | output_parser                            # Runnable 7
    )

    return chain, retriever
