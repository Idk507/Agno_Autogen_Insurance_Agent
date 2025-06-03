import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import chromadb
from chromadb.config import Settings
from config import AZURE_CONFIG

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
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=AZURE_CONFIG["embedding_deployment"],
                openai_api_version=AZURE_CONFIG["api_version"],
                azure_endpoint=AZURE_CONFIG["base_url"],
                openai_api_key=AZURE_CONFIG["api_key"]
            )
            self.llm = AzureChatOpenAI(
                azure_deployment=AZURE_CONFIG["gpt_deployment"],
                openai_api_version=AZURE_CONFIG["api_version"],
                azure_endpoint=AZURE_CONFIG["base_url"],
                openai_api_key=AZURE_CONFIG["api_key"],
                temperature=0.1
            )
        except Exception as e:
            print(f"Error initializing Azure clients: {e}")
            raise
    
    def setup_chromadb(self):
        """Initialize ChromaDB client for persistent storage"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
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
                print(f"No PDF documents found or error loading PDFs: {e}")
            
            try:
                from langchain_community.document_loaders import TextLoader
                text_files = list(Path(self.data_folder).glob("**/*.txt"))
                for text_file in text_files:
                    loader = TextLoader(str(text_file), encoding='utf-8')
                    text_docs = loader.load()
                    documents.extend(text_docs)
            except Exception as e:
                print(f"Error loading text files: {e}")
            
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                word_files = list(Path(self.data_folder).glob("**/*.docx"))
                for word_file in word_files:
                    loader = Docx2txtLoader(str(word_file))
                    word_docs = loader.load()
                    documents.extend(word_docs)
            except Exception as e:
                print(f"Error loading Word files: {e}")
            
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
            return []
        except Exception as e:
            print(f"Error processing documents: {e}")
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
                self.vectorstore.persist()
        except Exception as e:
            print(f"Error creating/loading vectorstore: {e}")
            raise
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        try:
            if not self.vectorstore:
                return "No knowledge base available."
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            if not results:
                return "No relevant information found in the knowledge base."
            context_parts = [f"[Relevance: {1-score:.2f}] {doc.page_content}" for doc, score in results]
            return "\n\n".join(context_parts)
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "Error retrieving information from knowledge base."
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search documents and return detailed results"""
        try:
            if not self.vectorstore:
                return []
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return [{
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": 1 - score,
                "source": doc.metadata.get("source", "Unknown")
            } for doc, score in results]
        except Exception as e:
            print(f"Error searching documents: {e}")
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
                "current_collection": self.collection_name,
                "document_count": self.chroma_client.get_collection(self.collection_name).count() if self.collection_name in collection_names else 0
            }
            return stats
        except Exception as e:
            return {"error": str(e)}
    
    def reset_vectorstore(self):
        """Reset/delete the vectorstore"""
        try:
            if self.collection_name in [col.name for col in self.chroma_client.list_collections()]:
                self.chroma_client.delete_collection(self.collection_name)
            self.vectorstore = None
        except Exception as e:
            print(f"Error resetting vectorstore: {e}")