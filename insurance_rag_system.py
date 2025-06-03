import os
import asyncio
import json
from typing import List, Dict, Any
from pathlib import Path

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
from langchain.schema import Document

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Azure Configuration - Updated format
AZURE_CONFIG = {
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
        self.setup_azure_clients()
        self.setup_chromadb()
        
    def setup_azure_clients(self):
        """Initialize Azure OpenAI clients"""
        try:
            # Initialize embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_CONFIG["embedding_deployment"],
                openai_api_version=AZURE_CONFIG["api_version"],
                azure_endpoint=AZURE_CONFIG["base_url"],  # LangChain still uses azure_endpoint
                openai_api_key=AZURE_CONFIG["api_key"]
            )
            
            # Initialize LLM
            self.llm = AzureChatOpenAI(
                azure_deployment=AZURE_CONFIG["gpt_deployment"],
                openai_api_version=AZURE_CONFIG["api_version"],
                azure_endpoint=AZURE_CONFIG["base_url"],  # LangChain still uses azure_endpoint
                openai_api_key=AZURE_CONFIG["api_key"],
                temperature=0.1
            )
            print(" Azure OpenAI clients initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Azure clients: {e}")
            raise
    
    def setup_chromadb(self):
        """Initialize ChromaDB client for persistent storage"""
        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print(f" ChromaDB client initialized with persistence at: {self.persist_directory}")
            
        except Exception as e:
            print(f" Error initializing ChromaDB: {e}")
            raise
    
    def load_and_process_documents(self) -> List[Document]:
        """Load and process PDF documents from data folder"""
        try:
            documents = []
            
            # Try to load PDF documents first
            try:
                pdf_loader = DirectoryLoader(
                    self.data_folder,
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                pdf_documents = pdf_loader.load()
                documents.extend(pdf_documents)
                print(f" Loaded {len(pdf_documents)} PDF documents")
            except Exception as e:
                print(f" No PDF documents found or error loading PDFs: {e}")
            
            # Also load text files as fallback
            try:
                from langchain_community.document_loaders import TextLoader
                text_files = list(Path(self.data_folder).glob("**/*.txt"))
                for text_file in text_files:
                    loader = TextLoader(str(text_file), encoding='utf-8')
                    text_docs = loader.load()
                    documents.extend(text_docs)
                print(f"Loaded {len(text_files)} text documents")
            except Exception as e:
                print(f"Error loading text files: {e}")
            
            # Load Word documents
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                word_files = list(Path(self.data_folder).glob("**/*.docx"))
                for word_file in word_files:
                    loader = Docx2txtLoader(str(word_file))
                    word_docs = loader.load()
                    documents.extend(word_docs)
                print(f" Loaded {len(word_files)} Word documents")
            except Exception as e:
                print(f" Error loading Word files: {e}")
            
            if not documents:
                print(" No documents found. Creating sample documents...")
                # Create sample insurance document if no documents found
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
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                split_docs = text_splitter.split_documents(documents)
                print(f" Processed {len(documents)} documents into {len(split_docs)} chunks")
                return split_docs
            else:
                print(" No documents could be loaded")
                return []
            
        except Exception as e:
            print(f"Error processing documents: {e}")
            return []
    
    def create_or_load_vectorstore(self, documents: List[Document] = None, force_recreate: bool = False):
        """Create or load ChromaDB vectorstore with persistent storage"""
        try:
            # Check if collection already exists
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            collection_exists = self.collection_name in existing_collections
            
            if collection_exists and not force_recreate:
                print(f" Loading existing ChromaDB collection: {self.collection_name}")
                
                # Load existing vectorstore
                self.vectorstore = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                
                # Get collection info
                collection = self.chroma_client.get_collection(self.collection_name)
                doc_count = collection.count()
                print(f" Loaded existing vectorstore with {doc_count} documents")
                
            else:
                if collection_exists and force_recreate:
                    print(f"Recreating ChromaDB collection: {self.collection_name}")
                    self.chroma_client.delete_collection(self.collection_name)
                else:
                    print(f" Creating new ChromaDB collection: {self.collection_name}")
                
                if not documents:
                    print(" No documents provided, loading from data folder...")
                    documents = self.load_and_process_documents()
                
                if not documents:
                    raise ValueError("No documents available for vectorstore creation")
                
                # Create new vectorstore with documents
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                
                print(f" Created new vectorstore with {len(documents)} document chunks")
                
                # Persist the data
                self.vectorstore.persist()
                print("Vectorstore persisted to disk")
            
        except Exception as e:
            print(f" Error creating/loading vectorstore: {e}")
            raise
    
    def add_documents_to_vectorstore(self, new_documents: List[Document]):
        """Add new documents to existing vectorstore"""
        try:
            if not self.vectorstore:
                print(" No vectorstore exists. Creating new one...")
                self.create_or_load_vectorstore(new_documents)
                return
            
            # Add documents to existing vectorstore
            self.vectorstore.add_documents(new_documents)
            self.vectorstore.persist()
            
            print(f" Added {len(new_documents)} new document chunks to vectorstore")
            
        except Exception as e:
            print(f" Error adding documents to vectorstore: {e}")
            raise
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        try:
            if not self.vectorstore:
                return "No knowledge base available."
            
            # Use similarity search with score to get more relevant results
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            if not results:
                return "No relevant information found in the knowledge base."
            
            # Format context with relevance scores
            context_parts = []
            for doc, score in results:
                context_parts.append(f"[Relevance: {1-score:.2f}] {doc.page_content}")
            
            context = "\n\n".join(context_parts)
            return context
            
        except Exception as e:
            print(f" Error retrieving context: {e}")
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
            print(f" Error searching documents: {e}")
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
    
    def reset_vectorstore(self):
        """Reset/delete the vectorstore (useful for testing)"""
        try:
            if self.collection_name in [col.name for col in self.chroma_client.list_collections()]:
                self.chroma_client.delete_collection(self.collection_name)
                print(f" Deleted collection: {self.collection_name}")
            
            self.vectorstore = None
            print(" Vectorstore reset successfully")
            
        except Exception as e:
            print(f" Error resetting vectorstore: {e}")


class InsuranceMultiAgentSystem:
    """Multi-agent system for insurance queries with ChromaDB backend"""
    
    def __init__(self, rag_system: InsuranceRAGSystem):
        self.rag_system = rag_system
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize all specialized agents with corrected AutoGen configuration"""
        
        # Updated AutoGen configuration for Azure OpenAI - using base_url instead of azure_endpoint
        config_list = [{
            "model": AZURE_CONFIG["gpt_deployment"],
            "api_type": "azure",
            "base_url": AZURE_CONFIG["base_url"],  # Changed from azure_endpoint to base_url
            "api_key": AZURE_CONFIG["api_key"],
            "api_version": AZURE_CONFIG["api_version"]
        }]
        
        llm_config = {
            "config_list": config_list,
            "temperature": 0.1,
            "timeout": 60,
        }
        
        # Knowledge Retrieval Agent
        self.agents["retriever"] = ConversableAgent(
            name="KnowledgeRetriever",
            system_message="""You are a Knowledge Retrieval Agent specializing in insurance information.
            Your role is to search through insurance documents stored in ChromaDB and provide relevant context for queries.
            When you receive a query, retrieve the most relevant information from the knowledge base.
            Present information clearly and cite specific policy details when available.
            If information is not available in the knowledge base, clearly state this limitation.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        # Claims Processing Agent
        self.agents["claims_agent"] = ConversableAgent(
            name="ClaimsSpecialist",
            system_message="""You are a Claims Processing Specialist with expertise in insurance claims.
            Your responsibilities include:
            - Explaining claims processes and requirements
            - Helping customers understand their coverage
            - Providing guidance on documentation needed
            - Explaining timelines and next steps
            - Identifying potential issues or delays
            
            Always be empathetic and helpful, as customers may be dealing with stressful situations.
            Provide step-by-step guidance and explain complex processes in simple terms.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        # Policy Advisor Agent
        self.agents["policy_advisor"] = ConversableAgent(
            name="PolicyAdvisor",
            system_message="""You are a Policy Advisory Agent specializing in insurance policy guidance.
            Your expertise includes:
            - Explaining different types of insurance coverage
            - Helping customers understand policy terms and conditions
            - Providing recommendations based on customer needs
            - Explaining premiums, deductibles, and coverage limits
            - Comparing different insurance options
            
            Focus on education and helping customers make informed decisions.
            Always consider the customer's specific situation and needs.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        # Customer Service Agent
        self.agents["customer_service"] = ConversableAgent(
            name="CustomerService",
            system_message="""You are a Customer Service Agent providing general insurance support.
            Your role includes:
            - Greeting customers and understanding their needs
            - Routing questions to appropriate specialists
            - Providing general insurance information
            - Following up on complex issues
            - Ensuring customer satisfaction
            
            Maintain a friendly, professional tone and always prioritize customer needs.
            Coordinate with other agents to provide comprehensive support.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        # Supervisor Agent
        self.agents["supervisor"] = ConversableAgent(
            name="Supervisor",
            system_message="""You are a Supervisor Agent overseeing the insurance support team.
            Your responsibilities:
            - Coordinating between different specialist agents
            - Ensuring comprehensive and accurate responses
            - Making final decisions on complex cases
            - Summarizing information from multiple agents
            - Ensuring customer queries are fully addressed
            
            Review all agent responses for accuracy and completeness before providing final answers.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "").upper()
        )
    
    def setup_group_chat(self):
        """Setup group chat with all agents"""
        agents_list = list(self.agents.values())
        
        self.group_chat = GroupChat(
            agents=agents_list,
            messages=[],
            max_round=10,
            speaker_selection_method="auto"
        )
        
        # Updated manager configuration with base_url
        manager_config_list = [{
            "model": AZURE_CONFIG["gpt_deployment"],
            "api_type": "azure",
            "base_url": AZURE_CONFIG["base_url"],  # Changed from azure_endpoint to base_url
            "api_key": AZURE_CONFIG["api_key"],
            "api_version": AZURE_CONFIG["api_version"]
        }]
        
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": manager_config_list,
                "temperature": 0.1
            }
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a customer query through the multi-agent system"""
        try:
            # Retrieve relevant context from ChromaDB
            context = self.rag_system.retrieve_context(query)
            search_results = self.rag_system.search_documents(query, k=3)
            
            # Enhanced query with context
            enhanced_query = f"""
            Customer Query: {query}
            
            Relevant Information from ChromaDB Knowledge Base:
            {context}
            
            Please provide a comprehensive response using the available information and your expertise.
            """
            
            # Setup group chat if not already done
            if not self.group_chat:
                self.setup_group_chat()
            
            # Start the conversation
            chat_result = self.agents["customer_service"].initiate_chat(
                self.manager,
                message=enhanced_query,
                clear_history=True
            )
            
            # Store conversation in memory
            self.rag_system.memory.chat_memory.add_user_message(query)
            
            # Extract the final response
            if chat_result and hasattr(chat_result, 'chat_history'):
                final_response = chat_result.chat_history[-1].get("content", "")
                self.rag_system.memory.chat_memory.add_ai_message(final_response)
            else:
                final_response = "I apologize, but I'm having trouble processing your request right now."
            
            return {
                "query": query,
                "response": final_response,
                "context_used": bool(context.strip()),
                "agents_involved": list(self.agents.keys()),
                "search_results": search_results,
                "vectorstore_stats": self.rag_system.get_collection_stats()
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": query,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "context_used": False,
                "agents_involved": [],
                "search_results": [],
                "vectorstore_stats": self.rag_system.get_collection_stats()
            }

    def simple_query(self, query: str) -> str:
        """Process a simple query without multi-agent complexity (fallback method)"""
        try:
            # Get context from ChromaDB
            context = self.rag_system.retrieve_context(query)
            
            # Use the LLM directly with context
            from langchain.schema import HumanMessage
            
            prompt = f"""
            You are an insurance expert assistant. Please answer the following question using the provided context from our ChromaDB knowledge base.
            
            Question: {query}
            
            Context from insurance documents:
            {context}
            
            Please provide a helpful, accurate response based on the context provided. If the context doesn't contain enough information, acknowledge this and provide general guidance.
            """
            
            response = self.rag_system.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            print(f" Error in simple query processing: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"



def main():
    """Main function to run the insurance multi-agent RAG application with ChromaDB"""
    print(" Starting Insurance Multi-Agent RAG System with ChromaDB...")
    
    try:
        # Initialize RAG system with ChromaDB
        print("\n Initializing RAG system with ChromaDB...")
        rag_system = InsuranceRAGSystem(
            data_folder="data",
            persist_directory="chroma_db"
        )
        
        # Check if vectorstore already exists or needs to be created
        stats = rag_system.get_collection_stats()
        print(f"\n ChromaDB Stats: {stats}")
        
        if stats.get("document_count", 0) == 0:
            print("\n No existing documents found. Loading and processing documents...")
            documents = rag_system.load_and_process_documents()
            
            if documents:
                print("\n Creating ChromaDB vectorstore...")
                rag_system.create_or_load_vectorstore(documents)
            else:
                print(" No documents loaded. System will work with limited knowledge.")
                rag_system.create_or_load_vectorstore()  # Create with sample data
        else:
            print(f"\nLoading existing ChromaDB collection with {stats['document_count']} documents...")
            rag_system.create_or_load_vectorstore()
        
        # Initialize multi-agent system
        print("\n Setting up multi-agent system...")
        agent_system = InsuranceMultiAgentSystem(rag_system)
        
        print("\n System initialized successfully!")
        print("\n" + "="*60)
        print("INSURANCE MULTI-AGENT RAG SYSTEM WITH CHROMADB")
        print("="*60)
        
        # Display ChromaDB information
        final_stats = rag_system.get_collection_stats()
        print(f"\n ChromaDB Information:")
        print(f"   • Persist Directory: {final_stats['persist_directory']}")
        print(f"   • Collection: {final_stats['current_collection']}")
        print(f"   • Documents: {final_stats['document_count']}")
        
        # Example queries for testing
        test_queries = [
            "What types of auto insurance coverage should I consider?",
            "How do I file a home insurance claim after storm damage?",
            "What's the difference between term and whole life insurance?",
        ]
        
        print("\nTesting with sample queries using ChromaDB backend...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n Query {i}: {query}")
            print("-" * 50)
            
            try:
                # Try simple approach first for demonstration
                response = agent_system.simple_query(query)
                print(f" Response: {response}")
                
                # Show search results from ChromaDB
                search_results = rag_system.search_documents(query, k=2)
                if search_results:
                    print(f"\n🔍 ChromaDB Search Results:")
                    for j, result in enumerate(search_results[:2], 1):
                        print(f"   {j}. Relevance: {result['relevance_score']:.2f}")
                        print(f"      Source: {result['source']}")
                        print(f"      Content: {result['content'][:100]}...")
                
            except Exception as e:
                print(f" Error processing query: {e}")
            
            # Add delay between queries
            import time
            time.sleep(1)
        
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Ask your insurance questions!")
        print("Special commands:")
        print("  'stats' - Show ChromaDB statistics")
        print("  'search [query]' - Search documents in ChromaDB")
        print("  'reload' - Reload documents from data folder")
        print("  'reset' - Reset ChromaDB collection")
        print("  'quit' - Exit")
        print("="*60)
        
        while True:
            user_input = input(f"\n Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n Thank you for using the Insurance Multi-Agent RAG System!")
                break
            
            if not user_input:
                print("Please enter a valid question or command.")
                continue
            
            # Handle special commands
            if user_input.lower() == 'stats':
                stats = rag_system.get_collection_stats()
                print(f"\n ChromaDB Statistics:")
                for key, value in stats.items():
                    print(f"   • {key}: {value}")
                continue
            
            if user_input.lower().startswith('search '):
                search_query = user_input[7:]  # Remove 'search ' prefix
                print(f"\n🔍 Searching ChromaDB for: '{search_query}'")
                results = rag_system.search_documents(search_query, k=5)
                
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. Relevance Score: {result['relevance_score']:.3f}")
                        print(f"   Source: {result['source']}")
                        print(f"   Content: {result['content'][:200]}...")
                        print("-" * 40)
                else:
                    print("No results found.")
                continue
            
            if user_input.lower() == 'reload':
                print("\n Reloading documents...")
                documents = rag_system.load_and_process_documents()
                if documents:
                    rag_system.create_or_load_vectorstore(documents, force_recreate=True)
                    print(" Documents reloaded successfully!")
                else:
                    print("No documents found to reload.")
                continue
            
            if user_input.lower() == 'reset':
                confirm = input(" This will delete all stored documents. Continue? (yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    rag_system.reset_vectorstore()
                    print("ChromaDB collection reset successfully!")
                continue
            
            print("\n Processing your query...")
            
            try:
                # Use simple method for faster response
                response = agent_system.simple_query(user_input)
                print(f"\n Response:")
                print("-" * 40)
                print(response)
                print("-" * 40)
                
                # Show relevant search results
                search_results = rag_system.search_documents(user_input, k=2)
                if search_results:
                    print(f"\n Related Information from ChromaDB:")
                    for i, result in enumerate(search_results, 1):
                        print(f"   {i}. Score: {result['relevance_score']:.2f} | {result['source']}")
                
            except Exception as e:
                print(f" Error: {e}")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nSystem Error: {e}")
        print("Please check your configuration and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
