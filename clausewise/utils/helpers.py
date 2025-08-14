import re
import nltk
import os
import requests
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Load environment variables
load_dotenv()
GRANITE_API_KEY = os.getenv("IBM_GRANITE_API_KEY")
GRANITE_URL = os.getenv("IBM_GRANITE_URL")

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MODEL_PATH = "ibm-granite/granite-3.3-2b-instruct"
HF_TOKEN = "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"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer once
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    token=HF_TOKEN
)

def segment_clauses(text):
    pattern = r'(\d+\.\d+|\bSection\s+\d+|\([a-z]\))'
    splits = [m.start() for m in re.finditer(pattern, text)]
    clauses = []
    if not splits:
        return nltk.sent_tokenize(text)
    for i in range(len(splits)):
        start = splits[i]
        end = splits[i+1] if i+1 < len(splits) else len(text)
        clauses.append(text[start:end].strip())
    return clauses

def simplify_clause(clause):
    conv = [{"role": "user", "content": f"Simplify this legal clause: {clause}"}]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(DEVICE)
    set_seed(42)
    output = model.generate(
        **input_ids,
        max_new_tokens=512,
    )
    prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prediction.strip()

def extract_entities(text):
    conv = [{"role": "user", "content": f"Extract named entities (parties, dates, obligations, monetary values, legal terms) from this legal text: {text}"}]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(DEVICE)
    set_seed(42)
    output = model.generate(
        **input_ids,
        max_new_tokens=512,
    )
    prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prediction.strip()

def classify_document_type(text):
    conv = [{"role": "user", "content": f"Classify this legal document into one of the following types: NDA, Lease, Employment Contract, Service Agreement. Text: {text}"}]
    input_ids = tokenizer.apply_chat_template(conv, return_tensors="pt", return_dict=True, add_generation_prompt=True).to(DEVICE)
    set_seed(42)
    output = model.generate(
        **input_ids,
        max_new_tokens=128,
    )
    prediction = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return prediction.strip()
