import httpx, openai
from openai import OpenAI
print("httpx:", httpx.__version__)
client = OpenAI()
print("✅ OpenAI client OK")
