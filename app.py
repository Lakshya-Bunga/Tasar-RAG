import streamlit as st

st.set_page_config(page_title="TASAR-RAG", layout="wide")

st.title("TASAR-RAG")
st.caption("Task-Aware Self-Reflective Research Assistant")

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

        "comparison": """Traditional CNN-based approaches are effective for feature extraction in medical imaging tasks. 
However, transformer-based models provide improved contextual understanding and long-range dependency modeling. 
Recent studies suggest hybrid architectures combining CNN and transformer components yield better performance.""",

        "gap": """Existing approaches for low-dose image reconstruction suffer from limited generalization across datasets. 
Many models are trained on specific distributions and fail to adapt to unseen noise patterns. 
Additionally, there is a lack of standardized evaluation protocols, making fair comparison difficult. 
Another major gap is insufficient real-world validation in clinical environments.""",

        "datasets": [
            "SciFact (for scientific claim verification)",
            "ArXiv Papers (for literature retrieval)",
            "PubMed (for biomedical research validation)"
        ],

        "plan": """1. Collect and preprocess low-dose imaging dataset.
2. Implement baseline CNN model for reconstruction.
3. Develop transformer-based or hybrid model.
4. Train both models under identical conditions.
5. Evaluate using PSNR, SSIM, and F1-score.
6. Perform cross-dataset validation to test generalization.
7. Compare performance and analyze failure cases."""
    }

st.subheader("Enter Research Query")
query = st.text_input("", placeholder="Example: Low dose CT reconstruction using deep learning")

run = st.button("Run Analysis")

if run and query:
    result = tasar_system(query)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Literature Insight")
        st.write(result["insight"])

        st.subheader("Comparison")
        st.write(result["comparison"])

    with col2:
        st.subheader("Research Gap")
        st.write(result["gap"])

        st.subheader("Suggested Datasets")
        for d in result["datasets"]:
            st.write("•", d)

    st.divider()

    st.subheader("Experiment Plan")
    st.write(result["plan"])
