import streamlit as st

st.set_page_config(page_title="TASAR-RAG", layout="centered")

st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #000000, #0f172a, #020617);
    color: white;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
    font-size: 15px;
}

/* Cards */
.card {
    background: rgba(255, 255, 255, 0.04);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin-bottom: 20px;
    border: 1px solid rgba(96, 165, 250, 0.15);
}

/* Card Title */
.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #e5e7eb;
}

/* Button */
div.stButton > button {
    background: linear-gradient(90deg, #1f2933, #374151);
    color: white;
    border-radius: 10px;
    padding: 8px 20px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">TASAR-RAG</div>', unsafe_allow_html=True)
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

        "comparison": """CNN-based methods are strong in capturing local features in medical images. 
Transformer-based architectures improve global context understanding and long-range dependencies. 
Hybrid models combining CNN and transformers achieve superior performance.""",

        "gap": """Current research suffers from poor generalization across datasets and noise variations. 
Most models fail to adapt to unseen real-world conditions. 
There is also a lack of standardized evaluation benchmarks and limited clinical validation.""",

        "datasets": [
            "SciFact (Scientific verification)",
            "ArXiv Papers (Literature corpus)",
            "PubMed (Biomedical validation)"
        ],

        "plan": """1. Collect and preprocess dataset
2. Train CNN baseline model
3. Implement transformer-based model
4. Evaluate using PSNR, SSIM, and F1-score
5. Perform cross-dataset validation
6. Compare performance and analyze limitations"""
    }

query = st.text_input("Enter Research Query")

if st.button("Run Analysis"):
    if query:
        result = tasar_system(query)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Literature Insight</div>
            {result["insight"]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Research Gap</div>
            {result["gap"]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Comparison</div>
            {result["comparison"]}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Suggested Datasets</div>
            {"<br>".join(result["datasets"])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">Experiment Plan</div>
            {result["plan"].replace("\n", "<br>")}
        </div>
        """, unsafe_allow_html=True)
