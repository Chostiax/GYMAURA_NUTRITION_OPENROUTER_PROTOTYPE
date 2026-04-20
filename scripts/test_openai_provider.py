from src.llm_provider import chat_completion

result = chat_completion(
    provider="openai",
    system_prompt="You are a helpful assistant. Reply with exactly: 1;2;3;4",
    user_prompt="Test",
    temperature=0.0,
    max_tokens=20,
)

print(result["raw_output"])
print(result["usage"])
print(result["estimated_cost_usd"])