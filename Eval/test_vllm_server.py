from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="",
)

completion = client.chat.completions.create(
    model="qwen-4b-swaa",
    messages=[
        {"role": "user", "content": "Who are you?"},
    ],
)

print(completion.choices[0].message)