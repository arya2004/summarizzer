import ollama
response = ollama.generate(model='mistral:latest', prompt='Why is the sky blue?')
print(response['response'])