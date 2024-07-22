from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.


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
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "What are go routines in Go? How are they different than Virtual threads in c#?"
    }],
    model=model,
)

print("Chat completion results:")
print(chat_completion.choices[0].message.content)