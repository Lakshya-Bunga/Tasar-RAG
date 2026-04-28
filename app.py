import streamlit as st

st.set_page_config(page_title="TASAR-RAG", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #000000, #1e1b4b, #020617);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    background: linear-gradient(90deg, #ffffff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #c4b5fd;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.04);
    padding: 20px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    margin-bottom: 20px;
    border: 1px solid rgba(167, 139, 250, 0.25);
}

.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #ddd6fe;
}

p {
    margin-bottom: 12px;
    line-height: 1.6;
}

div.stButton > button {
    background: linear-gradient(90deg, #312e81, #6366f1);
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

def retrieve(query, k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

def clean_text(text):
    import re

    text = text.lower()

    # Remove keyword dumps
    text = re.sub(r'keywords.*', '', text)

    # Remove (a), (1), etc
    text = re.sub(r'\(\s*[a-z0-9]+\s*\)', '', text)

    # Fix spacing
    text = re.sub(r'\s+', ' ', text)

    # Fix punctuation spacing
    text = text.replace(" .", ".").replace(" ,", ",")
    text = text.replace(" ;", ";").replace(" :", ":")

    # Fix bracket spacing
    text = text.replace("( ", "(").replace(" )", ")")

    # Split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean + capitalize
    sentences = [s.strip().capitalize() for s in sentences if len(s.strip()) > 30]

    return " ".join(sentences)

def format_paragraphs(text):
    sentences = text.split(". ")

    paragraphs = []
    temp = []

    for s in sentences:
        temp.append(s)
        if len(temp) == 3:
            paragraph = ". ".join(temp)
            if not paragraph.endswith("."):
                paragraph += "."
            paragraphs.append(paragraph)
            temp = []

    if temp:
        paragraph = ". ".join(temp)
        if not paragraph.endswith("."):
            paragraph += "."
        paragraphs.append(paragraph)

    return "".join([f"<p>{p}</p>" for p in paragraphs])

def tasar_system(query):
    results = retrieve(query, k=5)

    cleaned = clean_text(results[0]) + " " + clean_text(results[1])

    return {
        "insight": format_paragraphs(cleaned),

        "comparison": format_paragraphs("""CNN-based models are effective in capturing spatial features and are widely used in image classification tasks. 
However, they struggle with modeling long-range dependencies. Transformer-based architectures leverage self-attention mechanisms to capture global context. 
Hybrid models combining CNN and transformers demonstrate improved robustness and performance across datasets."""),

        "gap": format_paragraphs("""Current approaches suffer from limited generalization across datasets and varying noise conditions. 
Many models fail in real-world environments due to overfitting. 
There is also a lack of standardized evaluation protocols and limited real-world validation."""),

        "datasets": [
            "SciFact – scientific claim verification",
            "ArXiv – research literature corpus",
            "PubMed – biomedical validation",
            "ImageNet – benchmark dataset"
        ],

        "plan": [
            "Data Collection: Gather relevant datasets.",
            "Preprocessing: Normalize and clean data.",
            "Baseline Model: Implement CNN architecture.",
            "Advanced Model: Implement Transformer model.",
            "Training: Train both models.",
            "Evaluation: Use accuracy, precision, recall.",
            "Cross-Dataset Testing: Evaluate generalization.",
            "Analysis: Compare performance and errors.",
            "Deployment: Assess real-world usability."
        ]
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
            <ol>
                {"".join([f"<li>{step}</li>" for step in result["plan"]])}
            </ol>
        </div>
        """, unsafe_allow_html=True)
