import streamlit as st
import os
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Core imports
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.schema import Document, HumanMessage

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Page configuration
st.set_page_config(
    page_title="Insurance AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 5px rgba(255,255,255,0.3);
    }
    
    .chat-container {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        min-height: 60vh;
        max-height: 70vh;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .user-message {
        background: #2a2a3a;
        border-left: 4px solid #4CAF50;
        color: #e0e0e0;
    }
    
    .ai-message {
        background: #2a2a3a;
        border-left: 4px solid #9c27b0;
        color: #e0e0e0;
    }
    
    .search-result {
        background: #3c3c4c;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #ff9800;
        color: #e0e0e0;
    }
    
    .sidebar .sidebar-content {
        background: #1e1e2e;
        color: #e0e0e0;
    }
    
    .right-sidebar {
        background: #1e1e2e;
        color: #e0e0e0;
        padding: 1rem;
    }
    
    .agent-card {
        background: #2a2a3a;
        color: #e0e0e0;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .agent-card:hover {
        background: #3c3c4c;
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: #2a2a3a;
        color: #e0e0e0;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 3px solid #2196f3;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .analytics-container {
        background: #2a2a3a;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    .stButton>button {
        background: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: #45a049;
        transform: translateY(-1px);
    }
    
    .stTextArea textarea {
        background: #2a2a3a;
        color: #e0e0e0;
        border: 1px solid #4CAF50;
        border-radius: 8px;
    }
    
    .stSelectbox select {
        background: #2a2a3a;
        color: #e0e0e0;
        border-radius: 8px;
    }
    
    .stCheckbox label {
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Azure Configuration
@st.cache_resource
def get_azure_config():
    return {
        "api_key": "7NCSBIYABACOGKn3Y",
    "base_url": "https://om/",  
    "api_version": "2025-01-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment": "gpt-4o"
    }

class InsuranceRAGSystem:
    """Insurance-specific RAG system with ChromaDB for offline document processing and retrieval"""
    
    def __init__(self, data_folder: str = "data", persist_directory: str = "chroma_db"):
        self.data_folder = data_folder
        self.persist_directory = persist_directory
        self.collection_name = "insurance_documents"
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chroma_client = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.azure_config = get_azure_config()
        self.setup_azure_clients()
        self.setup_chromadb()
        
    def setup_azure_clients(self):
        """Initialize Azure OpenAI clients"""
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.azure_config["embedding_deployment"],
                openai_api_version=self.azure_config["api_version"],
                azure_endpoint=self.azure_config["base_url"],
                openai_api_key=self.azure_config["api_key"]
            )
            self.llm = AzureChatOpenAI(
                azure_deployment=self.azure_config["gpt_deployment"],
                openai_api_version=self.azure_config["api_version"],
                azure_endpoint=self.azure_config["base_url"],
                openai_api_key=self.azure_config["api_key"],
                temperature=0.1
            )
        except Exception as e:
            st.error(f"Error initializing Azure clients: {e}")
            raise
    
    def setup_chromadb(self):
        """Initialize ChromaDB client for persistent storage"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def load_and_process_documents(self) -> List[Document]:
        """Load and process PDF documents from data folder"""
        try:
            documents = []
            try:
                pdf_loader = DirectoryLoader(
                    self.data_folder,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_documents = pdf_loader.load()
                documents.extend(pdf_documents)
            except Exception as e:
                st.warning(f"No PDF documents found: {e}")
            
            try:
                from langchain_community.document_loaders import TextLoader
                text_files = list(Path(self.data_folder).glob("**/*.txt"))
                for text_file in text_files:
                    loader = TextLoader(str(text_file), encoding='utf-8')
                    text_docs = loader.load()
                    documents.extend(text_docs)
            except Exception as e:
                st.warning(f"Error loading text files: {e}")
            
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                word_files = list(Path(self.data_folder).glob("**/*.docx"))
                for word_file in word_files:
                    loader = Docx2txtLoader(str(word_file))
                    word_docs = loader.load()
                    documents.extend(word_docs)
            except Exception as e:
                st.warning(f"Error loading Word files: {e}")
            
            if not documents:
                sample_doc = Document(
                    page_content="""
                    Sample Insurance Policy Information:
                    
                    Auto Insurance Coverage Types:
                    1. Liability Coverage - Covers damages to others
                    2. Collision Coverage - Covers damage to your vehicle in accidents
                    3. Comprehensive Coverage - Covers theft, vandalism, weather damage
                    4. Personal Injury Protection - Covers medical expenses
                    
                    Home Insurance Coverage:
                    1. Dwelling Coverage - Covers structure of your home
                    2. Personal Property Coverage - Covers belongings
                    3. Liability Coverage - Covers accidents on property
                    4. Additional Living Expenses - Covers temporary housing
                    
                    Life Insurance Types:
                    1. Term Life Insurance - Temporary coverage for specific period
                    2. Whole Life Insurance - Permanent coverage with cash value
                    3. Universal Life Insurance - Flexible premiums and death benefits
                    """,
                    metadata={"source": "sample_insurance_policy.txt", "type": "sample"}
                )
                documents = [sample_doc]
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                split_docs = text_splitter.split_documents(documents)
                return split_docs
            else:
                return []
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return []
    
    def create_or_load_vectorstore(self, documents: List[Document] = None, force_recreate: bool = False):
        """Create or load ChromaDB vectorstore with persistent storage"""
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            collection_exists = self.collection_name in existing_collections
            
            if collection_exists and not force_recreate:
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                collection = self.chroma_client.get_collection(self.collection_name)
                doc_count = collection.count()
                st.success(f"Loaded existing vectorstore with {doc_count} documents")
            else:
                if collection_exists and force_recreate:
                    self.chroma_client.delete_collection(self.collection_name)
                
                if not documents:
                    documents = self.load_and_process_documents()
                
                if not documents:
                    raise ValueError("No documents available for vectorstore creation")
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                st.success(f"Created new vectorstore with {len(documents)} document chunks")
                self.vectorstore.persist()
        except Exception as e:
            st.error(f"Error creating/loading vectorstore: {e}")
            raise
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        try:
            if not self.vectorstore:
                return "No knowledge base available."
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            if not results:
                return "No relevant information found in the knowledge base."
            
            context_parts = []
            for doc, score in results:
                context_parts.append(f"[Relevance: {1-score:.2f}] {doc.page_content}")
            context = "\n\n".join(context_parts)
            return context
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return "Error retrieving information from knowledge base."
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents and return detailed results"""
        try:
            if not self.vectorstore:
                return []
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            search_results = []
            for doc, score in results:
                search_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 1 - score,
                    "source": doc.metadata.get("source", "Unknown")
                })
            return search_results
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            if not self.chroma_client:
                return {"error": "ChromaDB client not initialized"}
            
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            stats = {
                "persist_directory": self.persist_directory,
                "collections": collection_names,
                "current_collection": self.collection_name
            }
            if self.collection_name in collection_names:
                collection = self.chroma_client.get_collection(self.collection_name)
                stats["document_count"] = collection.count()
            else:
                stats["document_count"] = 0
            return stats
        except Exception as e:
            return {"error": str(e)}

class InsuranceMultiAgentSystem:
    """Multi-agent system for insurance queries with ChromaDB backend"""
    
    def __init__(self, rag_system: InsuranceRAGSystem):
        self.rag_system = rag_system
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.agent_responses = {}
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize all specialized agents"""
        config_list = [{
            "model": self.rag_system.azure_config["gpt_deployment"],
            "api_type": "azure",
            "base_url": self.rag_system.azure_config["base_url"],
            "api_key": self.rag_system.azure_config["api_key"],
            "api_version": self.rag_system.azure_config["api_version"]
        }]
        llm_config = {
            "config_list": config_list,
            "temperature": 0.1,
            "timeout": 60,
        }
        self.agents["retriever"] = ConversableAgent(
            name="KnowledgeRetriever",
            system_message="""You are a Knowledge Retrieval Agent specializing in insurance information.
            Search through insurance documents and provide relevant context for queries.
            Present information clearly and cite specific policy details when available.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        self.agents["claims_agent"] = ConversableAgent(
            name="ClaimsSpecialist",
            system_message="""You are a Claims Processing Specialist with expertise in insurance claims.
            Help customers understand their coverage, documentation needed, and timelines.
            Always be empathetic and helpful with step-by-step guidance.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        self.agents["policy_advisor"] = ConversableAgent(
            name="PolicyAdvisor",
            system_message="""You are a Policy Advisory Agent specializing in insurance policy guidance.
            Explain different types of coverage, policy terms, and provide recommendations.
            Focus on education and helping customers make informed decisions.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        self.agents["customer_service"] = ConversableAgent(
            name="CustomerService",
            system_message="""You are a Customer Service Agent providing general insurance support.
            Greet customers, understand their needs, and coordinate with specialists.
            Maintain a friendly, professional tone and prioritize customer satisfaction.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
    
    def simple_query(self, query: str) -> Dict[str, Any]:
        """Process a simple query and track which agent responds"""
        try:
            agent_to_use = self.determine_agent(query)
            context = self.rag_system.retrieve_context(query)
            search_results = self.rag_system.search_documents(query, k=3)
            prompt = f"""
            You are an insurance expert assistant. Answer the question using the provided context.
            
            Question: {query}
            
            Context from insurance documents:
            {context}
            
            Provide a helpful, accurate response based on the context. If context is insufficient, 
            acknowledge this and provide general guidance.
            """
            response = self.rag_system.llm.invoke([HumanMessage(content=prompt)])
            return {
                "response": response.content,
                "agent_used": agent_to_use,
                "context_used": bool(context.strip()),
                "search_results": search_results,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "agent_used": "error_handler",
                "context_used": False,
                "search_results": [],
                "timestamp": datetime.now()
            }
    
    def determine_agent(self, query: str) -> str:
        """Determine which agent should handle the query based on keywords"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['claim', 'file', 'accident', 'damage', 'report']):
            return "ClaimsSpecialist"
        elif any(word in query_lower for word in ['policy', 'coverage', 'premium', 'deductible', 'compare']):
            return "PolicyAdvisor"
        elif any(word in query_lower for word in ['search', 'find', 'document', 'information']):
            return "KnowledgeRetriever"
        else:
            return "CustomerService"

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'agent_system' not in st.session_state:
    st.session_state.agent_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'show_agents' not in st.session_state:
    st.session_state.show_agents = True
if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = False

@st.cache_resource
def initialize_system():
    """Initialize the RAG and agent systems"""
    try:
        rag_system = InsuranceRAGSystem(
            data_folder="data",
            persist_directory="chroma_db"
        )
        stats = rag_system.get_collection_stats()
        if stats.get("document_count", 0) == 0:
            documents = rag_system.load_and_process_documents()
            rag_system.create_or_load_vectorstore(documents)
        else:
            rag_system.create_or_load_vectorstore()
        agent_system = InsuranceMultiAgentSystem(rag_system)
        return rag_system, agent_system, True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, False

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🏥 Insurance AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing AI Assistant..."):
            rag_system, agent_system, success = initialize_system()
            if success:
                st.session_state.rag_system = rag_system
                st.session_state.agent_system = agent_system
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system. Please check your configuration.")
                return
    
    # Layout with sidebars
    left_col, main_col, right_col = st.columns([1, 3, 1])
    with left_col:
        with st.sidebar:
            st.header("🕐 Recent Activity")
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history[-3:]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>{chat['timestamp'].strftime('%H:%M:%S')}</strong><br>
                        Agent: {chat['agent_used']}<br>
                        Query: {chat['query'][:30]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Start chatting to see recent activity!")
            
   
    # Right Sidebar: Agents and Analytics
    with right_col:
        with st.sidebar.container():
            st.header("🤖 Agents & Insights")
            st.session_state.show_agents = st.checkbox("Show Available Agents", value=True)
            
            if st.session_state.show_agents:
                agents = [
                    ("🔍 Knowledge Retriever", "Searches and retrieves information"),
                    ("📋 Claims Specialist", "Handles insurance claims"),
                    ("📝 Policy Advisor", "Provides policy guidance"),
                    ("🎯 Customer Service", "General support and routing")
                ]
                for agent_name, description in agents:
                    st.markdown(f"""
                    <div class="agent-card">
                        <strong>{agent_name}</strong><br>
                        <small>{description}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.session_state.show_analytics = st.checkbox("Show Analytics & Insights", value=False)
            
            if st.session_state.show_analytics:
                with st.container():
         
                    st.subheader("📈 Analytics & Insights")
                    if st.session_state.chat_history:
                        agent_usage = {}
                        for chat in st.session_state.chat_history:
                            agent = chat['agent_used']
                            agent_usage[agent] = agent_usage.get(agent, 0) + 1
                        
                        if agent_usage:
                            df_agents = pd.DataFrame(list(agent_usage.items()), columns=['Agent', 'Usage Count'])
                            colors = ['#FF6B6B', '#4CAF50', '#2196F3', '#FFC107']  # Distinct colors
                            fig = px.pie(
                                df_agents,
                                values='Usage Count',
                                names='Agent',
                                title='Agent Usage Distribution',
                                height=250,
                                template='plotly_dark',
                                color_discrete_sequence=colors
                            )
                            fig.update_traces(
                                textinfo='percent+label',
                                marker=dict(line=dict(color='#e0e0e0', width=1))
                            )
                            fig.update_layout(
                                margin=dict(t=30, b=10, l=10, r=10),
                                title_font_size=14,
                                font=dict(color='#e0e0e0')
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        context_usage = [1 if chat['context_used'] else 0 for chat in st.session_state.chat_history]
                        if context_usage:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                y=context_usage,
                                mode='lines+markers',
                                name='Context Used',
                                line=dict(color='#FF6B6B', width=2),
                                marker=dict(size=8, color='#FF6B6B')
                            ))
                            fig2.update_layout(
                                title='Knowledge Base Usage Over Time',
                                yaxis_title='Context Used (1=Yes, 0=No)',
                                xaxis_title='Message Number',
                                height=250,
                                template='plotly_dark',
                                margin=dict(t=30, b=10, l=10, r=10),
                                title_font_size=14,
                                font=dict(color='#e0e0e0'),
                                yaxis=dict(range=[-0.2, 1.2], tickvals=[0, 1])
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("Start chatting to see analytics!")
                    st.markdown('</div>', unsafe_allow_html=True)

     # Left Sidebar: Recent Activity and System Controls
    with left_col:
        with st.sidebar:
            
            if st.button("🔄 Refresh Knowledge Base"):
                with st.spinner("Refreshing..."):
                    documents = st.session_state.rag_system.load_and_process_documents()
                    st.session_state.rag_system.create_or_load_vectorstore(documents, force_recreate=True)
                    st.success("Knowledge base refreshed!")
                    st.rerun()
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()
    
    
    # Main Content: Chat Interface
    with main_col:
        st.header("💬 Chat with AI Assistant")
        with st.container():
            chat_container = st.container()
            with chat_container:
                for i, chat in enumerate(st.session_state.chat_history):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {chat['query']}
                        <small style="color: #888;">
                            {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <strong>🤖 {chat['agent_used']}:</strong> {chat['response']}
                        <br><small style="color: #888;">
                            Context used: {'Yes' if chat['context_used'] else 'No'} | 
                            Search results: {len(chat['search_results'])}
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
                    if chat['search_results']:
                        with st.expander(f"📚 Knowledge Base Results ({len(chat['search_results'])})"):
                            for j, result in enumerate(chat['search_results'][:3]):
                                st.markdown(f"""
                                <div class="search-result">
                                    <strong>Result {j+1}</strong> (Relevance: {result['relevance_score']:.2f})<br>
                                    <strong>Source:</strong> {result['source']}<br>
                                    <strong>Content:</strong> {result['content'][:200]}...
                                </div>
                                """, unsafe_allow_html=True)
        
        st.markdown("---")
        user_input = st.text_area(
            "Ask your insurance question:",
            placeholder="e.g., What types of auto insurance coverage should I consider?",
            height=100
        )
        col_send, col_example = st.columns([1, 1])
        with col_send:
            if st.button("📤 Send Message", type="primary"):
                if user_input.strip():
                    with st.spinner("🤔 Thinking..."):
                        result = st.session_state.agent_system.simple_query(user_input)
                        chat_entry = {
                            "query": user_input,
                            "response": result["response"],
                            "agent_used": result["agent_used"],
                            "context_used": result["context_used"],
                            "search_results": result["search_results"],
                            "timestamp": result["timestamp"]
                        }
                        st.session_state.chat_history.append(chat_entry)
                        st.rerun()
                else:
                    st.warning("Please enter a question first!")
        with col_example:
            example_queries = [
                "What types of auto insurance coverage should I consider?",
                "How do I file a home insurance claim?",
                "What's the difference between term and whole life insurance?"
            ]
            selected_example = st.selectbox(
                "Or try an example:",
                ["Select an example..."] + example_queries
            )
            if selected_example != "Select an example...":
                st.text_area(
                    "Ask your insurance question:",
                    value=selected_example,
                    height=100,
                    key="example_input"
                )

if __name__ == "__main__":
    main()