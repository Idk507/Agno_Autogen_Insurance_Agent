

import os
import json
import time
from typing import List, Dict, Any
from pathlib import Path

# Core imports
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import streamlit as st
from streamlit_chat import message  # Fixed import

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Azure Configuration
AZURE_CONFIG = {
    "api_key": "",
    "base_url": "https:///",
    "api_version": "2025-01-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment": "gpt-4o"
}

class InsuranceRAGSystem:
    """Enhanced Insurance RAG system with ChromaDB backend"""
    
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
        self.setup_azure_clients()
        self.setup_chromadb()
        self.create_or_load_vectorstore()
        
    def setup_azure_clients(self):
        """Initialize Azure OpenAI clients"""
        try:
            # Initialize embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_CONFIG["embedding_deployment"],
                openai_api_version=AZURE_CONFIG["api_version"],
                azure_endpoint=AZURE_CONFIG["base_url"],
                openai_api_key=AZURE_CONFIG["api_key"]
            )
            
            # Initialize LLM
            self.llm = AzureChatOpenAI(
                azure_deployment=AZURE_CONFIG["gpt_deployment"],
                openai_api_version=AZURE_CONFIG["api_version"],
                azure_endpoint=AZURE_CONFIG["base_url"],
                openai_api_key=AZURE_CONFIG["api_key"],
                temperature=0.1,
                max_retries=3,
                request_timeout=60
            )
            print(" Azure OpenAI clients initialized successfully")
            
        except Exception as e:
            print(f" Error initializing Azure clients: {e}")
            raise
    
    def setup_chromadb(self):
        """Initialize ChromaDB client for persistent storage"""
        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print(f" ChromaDB client initialized at: {self.persist_directory}")
            
        except Exception as e:
            print(f" Error initializing ChromaDB: {e}")
            raise
    
    def load_and_process_documents(self) -> List[Document]:
        """Load and process insurance documents"""
        try:
            documents = []
            
            # Load PDF documents
            pdf_files = list(Path(self.data_folder).glob("**/*.pdf"))
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load_and_split()
                    documents.extend(docs)
                    print(f" Loaded PDF: {pdf_file.name}")
                except Exception as e:
                    print(f" Error loading {pdf_file}: {e}")
            
            # Load text documents
            text_files = list(Path(self.data_folder).glob("**/*.txt"))
            for text_file in text_files:
                try:
                    loader = TextLoader(str(text_file), encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded text: {text_file.name}")
                except Exception as e:
                    print(f" Error loading {text_file}: {e}")
            
            # Load Word documents
            docx_files = list(Path(self.data_folder).glob("**/*.docx"))
            for docx_file in docx_files:
                try:
                    loader = Docx2txtLoader(str(docx_file))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f" Loaded DOCX: {docx_file.name}")
                except Exception as e:
                    print(f" Error loading {docx_file}: {e}")
            
            if not documents:
                print(" Creating sample insurance document")
                sample_doc = Document(
                    page_content="""
                    HLA Insurance Policy Information:
                    
                    HLA COMPLETEPROTECT:
                    - Enhanced protection policy for life's uncertainties
                    - Underwritten by Hong Leong Assurance Berhad
                    - Regulated by Bank Negara Malaysia
                    - Contact: 03-7650 1288 or www.hla.com.my
                    - Recommended to consult with HLA agent for personalized advice
                    - Multiple riders available for enhanced coverage
                    
                    HLA LIFE ESSENTIAL:
                    - Two coverage choices for individuals and businesses
                    - Builds strong foundation of protection for family
                    - Affordable and flexible coverage options
                    - Can enhance coverage as needs change
                    - Review Product Disclosure Sheet before purchasing
                    - Sales Illustration available for detailed terms
                    
                    GENERAL HLA POLICY FEATURES:
                    - Critical illness protection riders available
                    - Accidental death benefit options
                    - Investment-linked policy options
                    - Term life and whole life variants
                    - Premium payment flexibility
                    - Online policy management available
                    
                    CLAIMS PROCESS:
                    1. Contact HLA immediately after incident
                    2. Submit completed claim forms within 30 days
                    3. Provide required documentation
                    4. Claims assessment by HLA team
                    5. Settlement within 14 working days upon approval
                    
                    UNDERWRITING CONSIDERATIONS:
                    - Age and health status
                    - Occupation and lifestyle
                    - Sum assured requirements
                    - Medical examination may be required
                    - Financial underwriting for high sum assured
                    """,
                    metadata={"source": "HLA_policies_sample.txt", "type": "sample", "page": 0}
                )
                documents = [sample_doc]
            
            # Enhanced text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                separators=["\n\n•", "\n•", "\n\n", "\n", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(documents)
            print(f" Processed {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            print(f" Error processing documents: {e}")
            return []
    
    def create_or_load_vectorstore(self, documents: List[Document] = None, force_recreate: bool = False):
        """Create or load ChromaDB vectorstore"""
        try:
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            collection_exists = self.collection_name in existing_collections
            
            if collection_exists and not force_recreate:
                print(f" Loading existing collection: {self.collection_name}")
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                collection = self.chroma_client.get_collection(self.collection_name)
                print(f" Loaded {collection.count()} documents")
                
            else:
                if collection_exists:
                    print(f" Recreating collection: {self.collection_name}")
                    self.chroma_client.delete_collection(self.collection_name)
                
                if not documents:
                    documents = self.load_and_process_documents()
                
                if not documents:
                    print(" No documents available, creating sample knowledge base")
                    documents = self.load_and_process_documents()
                
                print(f" Creating new collection: {self.collection_name}")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    client=self.chroma_client,
                    collection_name=self.collection_name
                )
                print(f" Created vectorstore with {len(documents)} chunks")
            
        except Exception as e:
            print(f" Vectorstore error: {e}")
            # Create sample documents as fallback
            print(" Creating fallback sample knowledge base")
            sample_doc = Document(
                page_content="Fallback insurance knowledge base",
                metadata={"source": "fallback.txt", "type": "sample"}
            )
            self.vectorstore = Chroma.from_documents(
                documents=[sample_doc],
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.collection_name
            )
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context with metadata"""
        try:
            if not self.vectorstore:
                self.create_or_load_vectorstore()
                if not self.vectorstore:
                    return "Knowledge base not available"
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            context_parts = []
            for i, (doc, score) in enumerate(results, 1):
                metadata = doc.metadata
                source = metadata.get("source", "Unknown document")
                page = metadata.get("page", "N/A")
                relevance = 1 - score
                
                context_parts.append(
                    f" Source {i}: {source} (Page {page}, Relevance: {relevance:.2f})\n"
                    f"{doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}"
                )
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f" Retrieval error: {e}")
            return "Error retrieving information"
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents with detailed metadata"""
        try:
            if not self.vectorstore:
                return []
            
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                metadata = doc.metadata
                search_results.append({
                    "content": doc.page_content,
                    "metadata": metadata,
                    "relevance_score": 1 - score,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "N/A")
                })
            
            return search_results
            
        except Exception as e:
            print(f" Search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        try:
            stats = {
                "persist_directory": self.persist_directory,
                "collection": self.collection_name,
                "embedding_model": AZURE_CONFIG["embedding_deployment"]
            }
            
            if self.chroma_client:
                collections = self.chroma_client.list_collections()
                stats["collections"] = [col.name for col in collections]
                
                if self.collection_name in stats["collections"]:
                    collection = self.chroma_client.get_collection(self.collection_name)
                    stats["document_count"] = collection.count()
            
            return stats
            
        except Exception as e:
            return {"error": str(e)}
    
    def reset_vectorstore(self):
        """Reset ChromaDB collection"""
        try:
            if self.chroma_client and self.collection_name in [col.name for col in self.chroma_client.list_collections()]:
                self.chroma_client.delete_collection(self.collection_name)
                print(f" Deleted collection: {self.collection_name}")
            
            self.vectorstore = None
            print(" Vectorstore reset")
            # Reinitialize after reset
            self.create_or_load_vectorstore()
            
        except Exception as e:
            print(f" Reset error: {e}")

class InsuranceMultiAgentSystem:
    """Enhanced multi-agent system with proper conversation flow"""
    
    def __init__(self, rag_system: InsuranceRAGSystem):
        self.rag_system = rag_system
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.conversation_history = []
        self.setup_agents()
        self.setup_group_chat()
    
    def setup_agents(self):
        """Initialize specialized insurance agents with enhanced functionality"""
        config_list = [{
            "model": AZURE_CONFIG["gpt_deployment"],
            "api_type": "azure",
            "base_url": AZURE_CONFIG["base_url"],
            "api_key": AZURE_CONFIG["api_key"],
            "api_version": AZURE_CONFIG["api_version"]
        }]
        
        llm_config = {
            "config_list": config_list,
            "temperature": 0.1,
            "timeout": 120,
            "cache_seed": 42
        }
        
        # Knowledge Retrieval Agent
        self.agents["retriever"] = ConversableAgent(
            name="KnowledgeRetriever",
            system_message="""You are an Insurance Knowledge Specialist. Your responsibilities:
            - Retrieve accurate policy information from knowledge base
            - Provide specific details about HLA policies with sources
            - Extract relevant policy features, benefits, and contact information
            - Always include source references (document name and page)
            - If information is not available, clearly state what's missing
            - Pass findings to PolicyAdvisor for customer presentation""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Policy Advisor Agent
        self.agents["policy_advisor"] = ConversableAgent(
            name="PolicyAdvisor",
            system_message="""You are a Licensed Policy Consultant. Your responsibilities:
            - Present HLA policy options clearly to customers
            - Explain coverage options, benefits, and limitations
            - Recommend suitable policies based on customer needs
            - Format information in customer-friendly language
            - Include contact information and next steps
            - Provide comparative analysis when multiple options exist
            - Pass complete response to ComplianceOfficer for review""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Claims Processing Agent
        self.agents["claims_agent"] = ConversableAgent(
            name="ClaimsSpecialist",
            system_message="""You are a Senior Claims Adjuster. Responsibilities:
            - Guide customers through claims process when relevant
            - Explain documentation requirements
            - Provide claims timeline and procedures
            - Only engage when query involves claims or procedures""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Customer Service Agent
        self.agents["customer_service"] = ConversableAgent(
            name="CustomerService",
            system_message="""You are the Primary Customer Interface. Responsibilities:
            - Analyze customer queries and route to appropriate specialists
            - For HLA policy recommendations: direct to KnowledgeRetriever first
            - Maintain professional and helpful tone
            - Ensure customer receives complete information
            - Coordinate with other agents as needed""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Compliance Agent
        self.agents["compliance_agent"] = ConversableAgent(
            name="ComplianceOfficer",
            system_message="""You are an Insurance Compliance Specialist. Responsibilities:
            - Review all policy information for accuracy
            - Ensure regulatory compliance with Malaysian insurance laws
            - Verify proper disclaimers and disclosures are included
            - Validate that recommendations include proper consultation advice
            - Approve final response before sending to Supervisor
            - Add compliance notes if needed""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Underwriting Agent
        self.agents["underwriting_agent"] = ConversableAgent(
            name="UnderwritingSpecialist",
            system_message="""You are a Senior Underwriter. Responsibilities:
            - Provide underwriting insights when relevant to query
            - Explain factors affecting policy eligibility and pricing
            - Only engage when query involves underwriting considerations""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1
        )
        
        # Supervisor Agent - FIXED
        self.agents["supervisor"] = ConversableAgent(
            name="Supervisor",
            system_message="""You are the Team Supervisor. Responsibilities:
            - Review the complete customer response from all agents
            - Ensure all aspects of the customer query are addressed
            - Format the final response professionally
            - Include source references and agent credits
            - State which agent primarily handled the query
            - Ensure response completeness before passing to RecommendationAgent
            - Your response should be the main answer the customer sees
            - Do NOT terminate - always pass to RecommendationAgent""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda msg: False  # Never terminate
        )
        
        # Recommendation Agent - FIXED
        self.agents["recommendation_agent"] = ConversableAgent(
            name="RecommendationAgent",
            system_message="""You are a Recommendation Specialist. Responsibilities:
            1. Provide ONE follow-up question the customer might ask next
            2. Provide ONE additional piece of useful information
            
            Format your response EXACTLY as:
            **Follow-up Suggestion**: [specific question related to the conversation]
            **Additional Insight**: [helpful information not yet covered]
            
            ALWAYS end your message with "TERMINATE" on a new line.
            Keep suggestions relevant and practical for the customer.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper()
        )
    
    def setup_group_chat(self):
        """Configure group chat workflow with proper speaker selection"""
        agent_list = [
            self.agents["customer_service"],
            self.agents["retriever"],
            self.agents["policy_advisor"],
            self.agents["claims_agent"],
            self.agents["underwriting_agent"],
            self.agents["compliance_agent"],
            self.agents["supervisor"],
            self.agents["recommendation_agent"]
        ]
        
        # Custom speaker selection function
        def custom_speaker_selection(last_speaker, groupchat):
            """Custom logic for agent selection"""
            messages = groupchat.messages
            if not messages:
                return self.agents["customer_service"]
            
            last_message = messages[-1]
            last_speaker_name = last_message.get("name", "")
            
            # Define the conversation flow
            if last_speaker_name == "CustomerService":
                return self.agents["retriever"]
            elif last_speaker_name == "KnowledgeRetriever":
                return self.agents["policy_advisor"]
            elif last_speaker_name == "PolicyAdvisor":
                return self.agents["compliance_agent"]
            elif last_speaker_name == "ComplianceOfficer":
                return self.agents["supervisor"]
            elif last_speaker_name == "Supervisor":
                return self.agents["recommendation_agent"]
            
            return None
        
        self.group_chat = GroupChat(
            agents=agent_list,
            messages=[],
            max_round=10,  # Reduced to prevent infinite loops
            speaker_selection_method=custom_speaker_selection,
            allow_repeat_speaker=False
        )
        
        manager_config = [{
            "model": AZURE_CONFIG["gpt_deployment"],
            "api_type": "azure",
            "base_url": AZURE_CONFIG["base_url"],
            "api_key": AZURE_CONFIG["api_key"],
            "api_version": AZURE_CONFIG["api_version"]
        }]
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={"config_list": manager_config, "temperature": 0.1}
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process insurance query through multi-agent system"""
        try:
            # Retrieve context before agent processing
            context = self.rag_system.retrieve_context(query)
            search_results = self.rag_system.search_documents(query, k=3)
            
            enhanced_query = f"""
Customer Query: {query}

Available Knowledge Base Context:
{context if context else 'No relevant context found'}

Instructions for Agent Team:
1. CustomerService: Route this HLA policy inquiry appropriately
2. KnowledgeRetriever: Extract relevant HLA policy information from knowledge base
3. PolicyAdvisor: Present policies clearly with benefits and contact info
4. ComplianceOfficer: Ensure regulatory compliance and proper disclosures
5. Supervisor: Create final formatted response with source citations
6. RecommendationAgent: Provide follow-up suggestions and TERMINATE

Please process systematically through each agent.
"""
            
            # Clear previous messages
            self.group_chat.messages = []
            
            # Initiate chat
            chat_result = self.agents["customer_service"].initiate_chat(
                self.manager,
                message=enhanced_query,
                clear_history=False,
                silent=False  # Set to True to reduce output during processing
            )
            
            # Extract responses
            chat_history = chat_result.chat_history if hasattr(chat_result, 'chat_history') else []
            
            final_response = ""
            recommendations = ""
            primary_agent = "CustomerService"
            all_agents = set()
            
            # Process all messages
            for msg in chat_history:
                agent_name = msg.get("name", "Unknown")
                all_agents.add(agent_name)
                content = msg.get("content", "")
                
                if agent_name == "Supervisor":
                    final_response = content
                    primary_agent = "PolicyAdvisor"  # Since they handle the main policy advice
                elif agent_name == "RecommendationAgent":
                    recommendations = content.replace("TERMINATE", "").strip()
            
            # Fallback if no supervisor response
            if not final_response and chat_history:
                # Find the most relevant response
                for msg in reversed(chat_history):
                    if msg.get("name") in ["PolicyAdvisor", "ComplianceOfficer", "Supervisor"]:
                        final_response = msg.get("content", "")
                        break
            
            # Create comprehensive response
            if not final_response:
                final_response = "I apologize, but I encountered an issue processing your request. Please try again."
            
            # Format final response
            complete_response = final_response
            if recommendations:
                complete_response += f"\n\n--- RECOMMENDATIONS ---\n{recommendations}"
            
            # Agent summary
            agent_list = sorted(list(all_agents))
            agents_involved = ", ".join(agent_list)
            
            return {
                "query": query,
                "response": complete_response,
                "primary_agent": primary_agent,
                "agents_involved": agents_involved,
                "knowledge_used": bool(context.strip()),
                "search_results": search_results,
                "chat_history": chat_history
            }
            
        except Exception as e:
            print(f" Processing error: {e}")
            return {
                "query": query,
                "response": f"I apologize, but I encountered a system error while processing your request: {str(e)}",
                "primary_agent": "Error",
                "agents_involved": "Error Handler",
                "knowledge_used": False,
                "search_results": [],
                "chat_history": []
            }

# Initialize systems with caching
@st.cache_resource(show_spinner=False)
def initialize_systems():
    """Initialize RAG and MultiAgent systems with caching"""
    with st.spinner("🧠 Initializing insurance knowledge system..."):
        rag_system = InsuranceRAGSystem(
            data_folder="data",
            persist_directory="chroma_db"
        )
        
        # Check existing vectorstore
        stats = rag_system.get_collection_stats()
        doc_count = stats.get("document_count", 0)
        
        if doc_count < 10:
            rag_system.create_or_load_vectorstore()
    
    with st.spinner("🤖 Starting insurance agent team..."):
        agent_system = InsuranceMultiAgentSystem(rag_system)
    
    return rag_system, agent_system

def display_system_info(rag_system):
    """Display system configuration in sidebar"""
    st.sidebar.title("Insurance Agent System")
    st.sidebar.divider()
    
    
    # System status
    stats = rag_system.get_collection_stats()
    doc_count = stats.get("document_count", 0)
    st.sidebar.divider()
    st.sidebar.subheader("Knowledge Base Status")
    st.sidebar.metric("Documents in Knowledge Base", doc_count)
    st.sidebar.markdown(f"- **Collection**: `{stats.get('collection', '')}`")
    st.sidebar.markdown(f"- **Storage**: `{stats.get('persist_directory', '')}`")
    
    # Agent information
    st.sidebar.divider()
    st.sidebar.subheader("Specialized Agents")
    agents_info = [
        "👤 CustomerService - Primary interface",
        "🔍 KnowledgeRetriever - Document search",
        "📊 PolicyAdvisor - Policy recommendations", 
        "📝 ClaimsSpecialist - Claims guidance",
        "📈 UnderwritingSpecialist - Risk assessment",
        "✅ ComplianceOfficer - Regulatory compliance",
        "👨‍💼 Supervisor - Response coordination",
        "💡 RecommendationAgent - Follow-up suggestions"
    ]
    
    for agent in agents_info:
        st.sidebar.markdown(f"- {agent}")

def main():
    """Main Streamlit application"""
    # Set page config
    st.set_page_config(
        page_title="HLA Insurance Assistant",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize systems
    rag_system, agent_system = initialize_systems()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_steps" not in st.session_state:
        st.session_state.agent_steps = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Display system info in sidebar
    display_system_info(rag_system)
    
    # Main chat area
    st.title("HLA Insurance Assistant")
    st.caption("Powered by Azure OpenAI and specialized insurance agents")
    st.divider()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{msg['id']}")
            else:
                message(msg["content"], key=f"assistant_{msg['id']}")
    
    # User input
    user_input = st.chat_input("Ask about HLA insurance policies...")
    
    # Process user input
    if user_input and not st.session_state.processing:
        st.session_state.processing = True
        
        # Add user message to chat
        user_msg_id = f"msg_{int(time.time()*1000)}"
        st.session_state.messages.append({"id": user_msg_id, "role": "user", "content": user_input})
        
        # Process with agent system
        with st.spinner("🤖 Consulting with our insurance specialists..."):
            result = agent_system.process_query(user_input)
            
            # Store agent steps
            st.session_state.agent_steps = result.get("chat_history", [])
            
            # Add assistant response to chat
            assistant_msg_id = f"msg_{int(time.time()*1000)}"
            st.session_state.messages.append({
                "id": assistant_msg_id,
                "role": "assistant",
                "content": result["response"],
                "primary_agent": result["primary_agent"],
                "agents_involved": result["agents_involved"],
                "knowledge_used": result["knowledge_used"],
                "search_results": result.get("search_results", [])
            })
        
        st.session_state.processing = False
        st.rerun()
    
    # Display agent processing steps in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Agent Processing")
    
    if st.session_state.agent_steps:
        agent_icons = {
            "CustomerService": "👤",
            "KnowledgeRetriever": "🔍",
            "PolicyAdvisor": "📊",
            "ClaimsSpecialist": "📝",
            "UnderwritingSpecialist": "📈",
            "ComplianceOfficer": "✅",
            "Supervisor": "👨‍💼",
            "RecommendationAgent": "💡"
        }
        
        for i, step in enumerate(st.session_state.agent_steps):
            agent_name = step.get("name", "Unknown")
            icon = agent_icons.get(agent_name, "🤖")
            
            with st.sidebar.expander(f"{icon} Step {i+1}: {agent_name}", expanded=(i == len(st.session_state.agent_steps)-1)):
                content = step.get("content", "")
                st.markdown(f"**Agent**: {agent_name}")
                st.markdown(f"**Response**:\n{content}")
    else:
        st.sidebar.info("No agent steps recorded yet. Ask a question to see the processing flow.")
    
    # System commands in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("System Commands")
    
    if st.sidebar.button("Show Knowledge Base Stats"):
        stats = rag_system.get_collection_stats()
        st.sidebar.json(stats, expanded=False)
    
    search_query = st.sidebar.text_input("Search knowledge base")
    if search_query:
        with st.spinner(f"🔍 Searching for '{search_query}'..."):
            results = rag_system.search_documents(search_query, k=3)
        if results:
            st.sidebar.subheader(f"Search Results ({len(results)})")
            for i, res in enumerate(results, 1):
                with st.sidebar.expander(f"Result {i} (Rel: {res['relevance_score']:.2f})"):
                    st.markdown(f"**Source**: {res.get('source', 'Unknown')}")
                    st.markdown(f"**Page**: {res.get('page', 'N/A')}")
                    st.markdown(f"**Content**:\n{res['content'][:300]}...")
        else:
            st.sidebar.warning("No results found")
    
    if st.sidebar.button("Reload Documents"):
        with st.spinner("🔄 Reloading documents..."):
            rag_system.create_or_load_vectorstore(force_recreate=True)
        st.sidebar.success("Documents reloaded successfully!")
    
    if st.sidebar.button("Reset Knowledge Base"):
        if st.sidebar.checkbox("Confirm: Delete ALL documents?"):
            with st.spinner("♻️ Resetting knowledge base..."):
                rag_system.reset_vectorstore()
            st.sidebar.success("Knowledge base reset complete!")

if __name__ == "__main__":
    main()