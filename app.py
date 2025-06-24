import streamlit as st
from rag_swot_bot import load_documents, create_vectorstore, generate_swot
import os

st.set_page_config(page_title="SWOT RAG Bot", layout="centered")
st.title("💼 SWOT Analysis Bot with Groq LLM ⚡")

# Step 1: Upload PDFs
uploaded_files = st.file_uploader("Upload PDF Reports", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    # Save uploaded files to disk only once
    for file in uploaded_files:
        file_path = f"data/{file.name}"
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
    st.success("✅ Files uploaded successfully!")

    # Step 2: Create embeddings/vectorstore ONCE and save in session_state
    if 'vectorstore' not in st.session_state:
        with st.spinner("🧠 Creating embeddings and vectorstore..."):
            docs = load_documents("data")
            st.session_state.vectorstore = create_vectorstore(docs)
        st.success("✅ Embeddings and vectorstore ready!")

    # Step 3: Ask for Company Name AFTER vectorstore is ready
    company_name = st.text_input("Enter the Company Name for SWOT Analysis")

    if company_name:
        # Step 4: Generate SWOT Button enabled only when company_name is provided
        if st.button("🚀 Generate SWOT Analysis"):
            with st.spinner("Crunching docs..."):
                retriever = st.session_state.vectorstore.as_retriever()
                swot = generate_swot(retriever, company_name)
            st.markdown("### 📊 SWOT Analysis:")
            if "<think>" in swot and "</think>" in swot:
                # Extract thinking part and cleaned output
                think_start = swot.find("<think>") + len("<think>")
                think_end = swot.find("</think>")
                thinking = swot[think_start:think_end].strip()
                cleaned_swot = swot[think_end + len("</think>"):].strip()

                st.markdown(cleaned_swot)
                st.download_button("💾 Download SWOT", cleaned_swot, file_name="swot_analysis.md")

                with st.expander("🧠 Model's Thought Process"):
                    st.markdown(thinking)
            else:
                st.markdown(swot)
                st.download_button("💾 Download SWOT", swot, file_name="swot_analysis.md")
    else:
        st.warning("⚠️ Please enter a company name to generate a SWOT analysis.")

else:
    st.info("Please upload one or more PDF reports to begin.")