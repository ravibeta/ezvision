from azure.ai.foundry.agents import Agent, ConnectedAgent, AgentService
from azure.ai.foundry.agents.models import Message, AgentContext

# === 1. Standard Agent: Tracks last 5 messages ===
class TrackingAgent(Agent):
    def __init__(self, name):
        super().__init__(name=name)
        self.message_history = []

    def on_message(self, message: Message, context: AgentContext):
        self.message_history.append(message.content)
        self.message_history = self.message_history[-5:]  # Keep last 5
        print(f"[{self.name}] Tracking messages: {self.message_history}")
        return None  # Delegates to connected agents

# === 2. Connected Agent: Handles "How many" queries ===
class HowManyAgent(ConnectedAgent):
    def __init__(self, name):
        super().__init__(name=name)

    def can_handle(self, message: Message, context: AgentContext) -> bool:
        return message.content.strip().lower().startswith("how many")

    def on_message(self, message: Message, context: AgentContext):
        return f"[{self.name}] Answering count-related query: {message.content}"

# === 3. Connected Agent: Handles "How far" queries ===
class HowFarAgent(ConnectedAgent):
    def __init__(self, name):
        super().__init__(name=name)

    def can_handle(self, message: Message, context: AgentContext) -> bool:
        return message.content.strip().lower().startswith("how far")

    def on_message(self, message: Message, context: AgentContext):
        return f"[{self.name}] Answering distance-related query: {message.content}"

# === 4. Assemble Multi-Agent Workflow ===
def build_multi_agent_system():
    tracker = TrackingAgent(name="TrackerAgent")
    how_many = HowManyAgent(name="HowManyAgent")
    how_far = HowFarAgent(name="HowFarAgent")

    tracker.connect(how_many)
    tracker.connect(how_far)

    service = AgentService(root_agent=tracker)
    return service

# === 5. Simulate Conversation ===
if __name__ == "__main__":
    service = build_multi_agent_system()

    queries = [
        "Hello there!",
        "How many satellites are visible?",
        "How far is the nearest airport?",
        "What's the weather like?",
        "How many birds are in the image?",
        "How far can drones fly on one charge?"
    ]

    for q in queries:
        response = service.run(Message(content=q))
        print(f"User: {q}")
        print(f"Agent Response: {response}\n")
