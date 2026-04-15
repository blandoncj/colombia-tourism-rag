import streamlit as st
from rag_chain import build_chain, TEMAS_WIKIPEDIA

st.set_page_config(
    page_title="Guía Turística Colombia",
    page_icon="🇨🇴",
    layout="wide",
)

st.title("🇨🇴 Guía Turística de Colombia")
st.caption("""
    Pregúntame sobre destinos, lugares para visitar, historia y cultura
    colombiana.
""")

RETRIEVER_OPTIONS = {
    "Similarity (básico)": "similarity",
    "MMR — máxima diversidad": "mmr",
    "Multi-Query — múltiples preguntas": "multi_query",
    "Compression — extracción con LLM": "compression",
}

RETRIEVER_DESCRIPTIONS = {
    "similarity": "Busca los fragmentos más similares a la pregunta. Simple y rápido.",
    "mmr": "Maximum Marginal Relevance: recupera fragmentos diversos para evitar redundancia.",
    "multi_query": "El LLM genera varias reformulaciones de la pregunta y combina los resultados.",
    "compression": "El LLM extrae solo las oraciones relevantes de cada fragmento recuperado.",
}

# Mostrar los destinos disponibles en el sidebar
with st.sidebar:
    st.header("Destinos disponibles")
    for tema in TEMAS_WIKIPEDIA:
        st.markdown(f"- {tema}")
    st.divider()
    st.markdown("**Fuente:** Wikipedia en español")

    selected_label = st.selectbox(
        "Tipo de retriever",
        options=list(RETRIEVER_OPTIONS.keys()),
    )
    selected_type = RETRIEVER_OPTIONS[selected_label]
    st.caption(RETRIEVER_DESCRIPTIONS[selected_type])

# Reconstruir el pipeline si el retriever cambia
if st.session_state.get("retriever_type") != selected_type:
    st.session_state.retriever_type = selected_type
    st.session_state.pop("chain", None)
    st.session_state.pop("retriever", None)
    st.session_state.messages = []

# Construir el pipeline una vez por sesión (o cuando cambia el retriever)
if "chain" not in st.session_state:
    with st.spinner("Cargando información turística desde Wikipedia..."):
        st.session_state.chain, st.session_state.retriever = build_chain(selected_type)
    st.success("¡Guía lista! Puedes hacer tus preguntas.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Historial de mensajes
for msg in st.session_state.messages:
    icon = "🧑‍💼" if msg["role"] == "user" else "🇨🇴"
    with st.chat_message(msg["role"], avatar=icon):
        st.write(msg["content"])

# Input del usuario
if question := st.chat_input("¿Qué quieres saber sobre Colombia?"):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(question)

    with st.chat_message("assistant", avatar="🇨🇴"):
        with st.spinner("Consultando..."):
            answer = st.session_state.chain.invoke(question)
            compressed_docs = st.session_state.retriever.invoke(question)

        st.write(answer)

        with st.expander("📄 Fragmentos de Wikipedia usados como contexto"):
            if compressed_docs:
                for i, doc in enumerate(compressed_docs, 1):
                    source = doc.metadata.get(
                        "title", doc.metadata.get("source", "Wikipedia"))
                    st.markdown(f"**Fragmento {i}** — `{source}`")
                    st.write(doc.page_content)
                    st.divider()
            else:
                st.write("No se encontraron fragmentos relevantes.")

    st.session_state.messages.append({"role": "assistant", "content": answer})
