from app.llms.llm_manager import LLMManager
from app.planner_engine.planner_agent import PlannerAgent
from app.prompts.chat_agent import ChatAgentPrompt

from app.memory.conversation_history.local_memory import LocalMemory

class ChatService:
    def __init__(self):
        self.llm_manager = LLMManager()
        self.planner_agent = PlannerAgent()
        self.conversation_history = LocalMemory()
    
    async def reply(self, user_query: str, user_id: str, trip_id: str, conversation_id: str) -> str:
        self.conversation_history.add_message(conversation_id, "user", user_query)        
        
        rendered_prompt = ChatAgentPrompt.format()

        try:
            response = self.llm_manager.call_llm_with_retry(
                user_query=user_query,
                prompt = rendered_prompt,
                response_model=ChatAgentPrompt.response_model(),
                conversation_id=conversation_id,
                max_tokens=400
            )
        except Exception as e:
            raise e
        
        if response.ready_to_plan:
            try:
                facts = ",".join(response.collected_facts)
                initial_query = self.conversation_history.get_history(conversation_id)[0]["content"]
                query = f"{initial_query},{facts}"
                plan = await self.planner_agent.run(
                    user_query=query,
                    user_id=user_id,
                    trip_id=trip_id
                )
                self.conversation_history.add_message(conversation_id, "assistant", plan.answer)
                return plan.answer
            except Exception as e:
                raise ValueError(f"Error during trip planning: {e}")
        else:
            self.conversation_history.add_message(conversation_id, "assistant", response.answer)
            return response.answer
