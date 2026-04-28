import streamlit as st

st.set_page_config(page_title="TASAR-RAG", layout="wide")
st.title("🔬 TASAR-RAG: Research Assistant")

@st.cache_resource
def load_system():
    try:
        import faiss
        import pickle
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index("faiss_index.bin")
        with open("documents.pkl", "rb") as f:
            documents = pickle.load(f)
        return embedder, index, documents
    except Exception as e:
        return None, None, str(e)

embedder, index, documents = load_system()

if embedder is None:
    st.error(f"Error loading system: {documents}")
    st.stop()

def retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def tasar_system(query):
    results = retrieve(query, k=3)
    return {
        "insight": results[0],
        "comparison": "CNN-based models perform well, while transformer-based models improve contextual understanding.",
        "gap": "Limited generalization across datasets and lack of standardized evaluation.",
        "datasets": ["SciFact", "ArXiv", "PubMed"],
        "plan": "Train CNN vs Transformer models and evaluate using Accuracy and F1-score."
    }

query = st.text_input("Enter your research query")

if st.button("Run"):
    if not query:
        st.warning("Please enter a query")
    else:
        result = tasar_system(query)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Literature Insight")
            st.write(result["insight"])
            st.subheader("Comparison")
            st.write(result["comparison"])
        with col2:
            st.subheader("Research Gap")
            st.write(result["gap"])
            st.subheader("Datasets")
            for d in result["datasets"]:
                st.write(d)
        st.subheader("Experiment Plan")
        st.write(result["plan"])
