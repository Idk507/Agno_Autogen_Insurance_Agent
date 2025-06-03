from typing import Dict, Any
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager
from config import AZURE_CONFIG
from rag_system import InsuranceRAGSystem

class InsuranceMultiAgentSystem:
    """Multi-agent system for insurance queries with ChromaDB backend"""
    
    def __init__(self, rag_system: InsuranceRAGSystem):
        self.rag_system = rag_system
        self.agents = {}
        self.group_chat = None
        self.manager = None
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize all specialized agents"""
        config_list = [{
            "model": AZURE_CONFIG["gpt_deployment"],
            "api_type": "azure",
            "base_url": AZURE_CONFIG["base_url"],
            "api_key": AZURE_CONFIG["api_key"],
            "api_version": AZURE_CONFIG["api_version"]
        }]
        llm_config = {"config_list": config_list, "temperature": 0.1, "timeout": 60}
        
        self.agents["retriever"] = ConversableAgent(
            name="KnowledgeRetriever",
            system_message="""You are a Knowledge Retrieval Agent specializing in insurance information.
            Your role is to search through insurance documents stored in ChromaDB and provide relevant context for queries.
            Present information clearly and cite specific policy details when available.
            If information is not available, state this limitation.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        self.agents["claims_agent"] = ConversableAgent(
            name="ClaimsSpecialist",
            system_message="""You are a Claims Processing Specialist with expertise in insurance claims.
            Provide step-by-step guidance and explain complex processes in simple terms.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        self.agents["policy_advisor"] = ConversableAgent(
            name="PolicyAdvisor",
            system_message="""You are a Policy Advisory Agent specializing in insurance policy guidance.
            Focus on education and helping customers make informed decisions.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        self.agents["customer_service"] = ConversableAgent(
            name="CustomerService",
            system_message="""You are a Customer Service Agent providing general insurance support.
            Maintain a friendly, professional tone and coordinate with other agents.""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        self.agents["supervisor"] = ConversableAgent(
            name="Supervisor",
            system_message="""You are a Supervisor Agent overseeing the insurance support team.
            Summarize information and ensure queries are fully addressed. End with 'TERMINATE' when done.""",
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
        manager_config_list = [{
            "model": AZURE_CONFIG["gpt_deployment"],
            "api_type": "azure",
            "base_url": AZURE_CONFIG["base_url"],
            "api_key": AZURE_CONFIG["api_key"],
            "api_version": AZURE_CONFIG["api_version"]
        }]
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={"config_list": manager_config_list, "temperature": 0.1}
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a customer query through the multi-agent system"""
        try:
            context = self.rag_system.retrieve_context(query)
            search_results = self.rag_system.search_documents(query, k=3)
            enhanced_query = f"""
            Customer Query: {query}
            Relevant Information from ChromaDB Knowledge Base:
            {context}
            Please provide a comprehensive response using the available information and your expertise.
            """
            if not self.group_chat:
                self.setup_group_chat()
            chat_result = self.agents["customer_service"].initiate_chat(
                self.manager,
                message=enhanced_query,
                clear_history=True
            )
            chat_history = self.manager.groupchat.messages
            final_response = chat_history[-1].get("content", "").replace("TERMINATE", "").strip() if chat_history else "I apologize, but I'm having trouble processing your request right now."
            self.rag_system.memory.chat_memory.add_user_message(query)
            self.rag_system.memory.chat_memory.add_ai_message(final_response)
            return {
                "query": query,
                "response": final_response,
                "context_used": bool(context.strip()),
                "agents_involved": list(self.agents.keys()),
                "search_results": search_results,
                "vectorstore_stats": self.rag_system.get_collection_stats(),
                "chat_history": chat_history
            }
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": query,
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "context_used": False,
                "agents_involved": [],
                "search_results": [],
                "vectorstore_stats": self.rag_system.get_collection_stats(),
                "chat_history": []
            }