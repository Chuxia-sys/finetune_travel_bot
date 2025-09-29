from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

# Load model and tokenizer from Hugging Face
MODEL_NAME = "Chuxia-sys/my-finetuned-travel-bot"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

class Prompt(BaseModel):
    text: str
    max_tokens: int = 200

@app.post("/generate")
def generate(prompt: Prompt):
    output = generator(prompt.text, max_new_tokens=prompt.max_tokens)
    return {"generated_text": output[0]["generated_text"]}
