from openai import OpenAI
import config
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = config.OPENAI_API_KEY
openai_api_base = config.OPENAI_API_BASE

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "you will always reply starting by a random number"
    }, {
        "role": "user",
        "content": "What are green threads? how different from virtual threads"
    }, ],
    model=model,
)

print("Chat completion results:")
print(chat_completion.choices[0].message.content)