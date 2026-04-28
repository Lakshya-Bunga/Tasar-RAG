import streamlit as st

st.set_page_config(page_title="TASAR-RAG", layout="centered")

# ---------- STYLING ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 600;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #b0bec5;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">TASAR-RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Task-Aware Self-Reflective Research Assistant</div>', unsafe_allow_html=True)

# ---------- LOAD SYSTEM ----------
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
    st.error(f"Error: {documents}")
    st.stop()

# ---------- RETRIEVAL ----------
def retrieve(query, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# ---------- TASAR ----------
def tasar_system(query):
    results = retrieve(query, k=3)
    return {
        "insight": results[0],

        "comparison": """CNN-based methods are effective for local feature extraction. 
Transformer-based architectures capture global dependencies better. 
Hybrid models combining both approaches are emerging as state-of-the-art.""",

        "gap": """Current models lack robustness across diverse datasets. 
There is no standardized benchmarking, and many methods fail in real-world noisy conditions. 
Clinical validation remains limited.""",

        "datasets": [
            "SciFact (verification)",
            "ArXiv (literature)",
            "PubMed (biomedical validation)"
        ],

        "plan": """1. Prepare dataset
2. Train CNN baseline
3. Train Transformer model
4. Evaluate using PSNR, SSIM
5. Perform cross-dataset validation
6. Compare results and analyze errors"""
    }

# ---------- INPUT ----------
query = st.text_input("Enter Research Query")

if st.button("Run Analysis"):
    if query:
        result = tasar_system(query)

        # ---------- CARDS ----------
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
        
