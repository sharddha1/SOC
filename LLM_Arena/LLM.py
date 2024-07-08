
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

models = {
    "llama3-8b-8192":ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192"),
    "mixtral-8x7b-32768": ChatGroq(temperature=0, groq_api_key=groq_api_key , model_name="mixtral-8x7b-32768"),
    "gemma2-9b-it": ChatGroq(temperature=0, groq_api_key=groq_api_key , model_name="gemma2-9b-it"),
    "google/text-bison-001":GoogleGenerativeAI(model="models/text-bison-001", google_api_key=google_api_key),
    "gemini-pro":GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
}

def get_random_models():
    import random
    random.seed()
    return random.sample(list(models.keys()), 2)

def get_response_model(model, prompt):
    if(model == "llama3-8b-8192" or model=="mixtral-8x7b-32768" or model=="gemma2-9b-it"):
        response = models[model].invoke(prompt)
        return response.content
    else:
        response = models[model].invoke(prompt)
        return response