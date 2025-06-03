import streamlit as st
from rag_system import InsuranceRAGSystem
from multi_agent_system import InsuranceMultiAgentSystem

@st.cache_resource
def init_rag_system(_version):
    rag_system = InsuranceRAGSystem(data_folder="data", persist_directory="chroma_db")
    stats = rag_system.get_collection_stats()
    if stats.get("document_count", 0) == 0:
        documents = rag_system.load_and_process_documents()
        if documents:
            rag_system.create_or_load_vectorstore(documents)
        else:
            rag_system.create_or_load_vectorstore()
    else:
        rag_system.create_or_load_vectorstore()
    return rag_system

@st.cache_resource
def init_agent_system(_version, rag_system):
    return InsuranceMultiAgentSystem(rag_system)

def main():
    st.set_page_config(page_title="Insurance Assistant", layout="wide")
    st.title("🏠 Insurance Multi-Agent Assistant")
    st.markdown("""
    Welcome! Ask any insurance-related question, and our team of specialized agents will assist you.
    Use the sidebar to view knowledge base stats, search documents, reload, or reset the collection.
    """)
    
    if 'rag_version' not in st.session_state:
        st.session_state.rag_version = 0
    
    rag_system = init_rag_system(st.session_state.rag_version)
    agent_system = init_agent_system(st.session_state.rag_version, rag_system)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Options")
        option = st.selectbox("Choose an action", ["Ask a Question", "View Stats", "Search Documents", "Reload Documents", "Reset Collection"])
        
        if option == "View Stats":
            stats = rag_system.get_collection_stats()
            st.subheader("📊 ChromaDB Statistics")
            for key, value in stats.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        elif option == "Search Documents":
            search_query = st.text_input("Enter search query:")
            if st.button("Search"):
                with st.spinner("Searching..."):
                    results = rag_system.search_documents(search_query, k=5)
                if results:
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i} (Score: {result['relevance_score']:.3f})"):
                            st.write(f"**Source:** {result['source']}")
                            st.write(f"**Content:** {result['content']}")
                else:
                    st.info("No results found.")
        
        elif option == "Reload Documents":
            if st.button("Reload Documents"):
                with st.spinner("Reloading documents..."):
                    st.session_state.rag_version += 1
                    st.experimental_rerun()
        
        elif option == "Reset Collection":
            if st.button("Reset Collection"):
                if st.checkbox("Confirm reset"):
                    with st.spinner("Resetting collection..."):
                        rag_system.reset_vectorstore()
                        st.session_state.rag_version += 1
                        st.success("Collection reset successfully!")
                        st.experimental_rerun()
                else:
                    st.info("Please confirm the reset action.")
    
    # Main Content
    if option == "Ask a Question":
        query = st.text_input("Enter your insurance-related question:", placeholder="e.g., What types of auto insurance coverage should I consider?")
        if st.button("Submit Query"):
            if query:
                with st.spinner("Our agents are processing your query..."):
                    result = agent_system.process_query(query)
                st.subheader("✅ Final Response")
                st.success(result["response"])
                
                st.subheader("💬 Agent Conversation")
                for msg in result["chat_history"]:
                    sender = msg.get("name", "Unknown")
                    content = msg.get("content", "").replace("TERMINATE", "").strip()
                    color = "#e6f3ff" if sender != "Supervisor" else "#d4edda"
                    st.markdown(
                        f"<div style='background-color: {color}; padding: 15px; border-radius: 10px; margin: 5px 0;'><strong>{sender}:</strong> {content}</div>",
                        unsafe_allow_html=True
                    )
                
                st.subheader("🔍 Related Documents")
                for i, result in enumerate(result["search_results"], 1):
                    with st.expander(f"Document {i} (Score: {result['relevance_score']:.3f})"):
                        st.write(f"**Source:** {result['source']}")
                        st.write(f"**Content:** {result['content']}")
            else:
                st.warning("Please enter a question to proceed.")
    
    st.markdown("---")
    st.caption("Powered by Streamlit, AutoGen, and ChromaDB | © 2023 Insurance Assistant")

if __name__ == "__main__":
    main()