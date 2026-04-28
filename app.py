import streamlit as st

st.set_page_config(page_title="TASAR-RAG", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        background-color: #1e1e1e;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 TASAR-RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Task-Aware Self-Reflective Research Assistant</div>', unsafe_allow_html=True)

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

st.markdown("### Enter Research Query")
query = st.text_input("", placeholder="e.g., Low dose CT reconstruction using deep learning")

if st.button("Run Analysis"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Analyzing research..."):
            result = tasar_system(query)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Literature Insight")
            st.write(result["insight"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Comparison")
            st.write(result["comparison"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Research Gap")
            st.write(result["gap"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Suggested Datasets")
            for d in result["datasets"]:
                st.write(f"• {d}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Experiment Plan")
        st.write(result["plan"])
        st.markdown('</div>', unsafe_allow_html=True)
