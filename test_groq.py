
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

models = client.models.list()
print("Available models:")

for model in models.data:
    if getattr(model, "active", False):
        print(model.id)
