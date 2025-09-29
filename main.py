# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

# Initialize FastAPI
app = FastAPI(title="Travel Bot API", version="1.0")

# Global variables to hold model components
model = None
tokenizer = None
generator = None

MODEL_NAME = "Chuxia-sys/my-finetuned-travel-bot"

@app.on_event("startup")
async def load_model():
    global model, tokenizer, generator
    try:
        logging.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logging.info("Model loaded successfully!")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

class Prompt(BaseModel):
    text: str
    max_tokens: int = 200

@app.post("/generate")
async def generate(prompt: Prompt):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        output = generator(
            prompt.text,
            max_new_tokens=prompt.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        return {"generated_text": output[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
