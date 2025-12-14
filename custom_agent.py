"""
Custom Deep Agent with hybrid model setup:
- Main agent: Llama 3.1 8B (tool use)
- Thinking subagent: DeepSeek R1 (reasoning)
"""

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama

# Main agent with tool support
main_model = ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
)

# Thinking subagent (DeepSeek R1)
thinking_subagent = {
    "name": "thinker",
    "description": "Use for complex reasoning, planning, and problem-solving tasks that require deep thinking",
    "prompt": "You are a reasoning expert. Think step-by-step and show your thought process.",
    "model": ChatOllama(
        model="deepseek-r1:latest",
        base_url="http://localhost:11434",
    ),
}

# Create agent with hybrid setup
agent = create_deep_agent(
    model=main_model,
    subagents=[thinking_subagent],
    system_prompt="You are a helpful coding assistant. For complex problems, delegate to the 'thinker' subagent.",
)

if __name__ == "__main__":
    # Example usage
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Design a scalable architecture for a real-time chat application"
        }]
    })
    print(result)
