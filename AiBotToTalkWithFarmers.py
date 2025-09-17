from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import torch, logging


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)


logging.getLogger("accelerate").setLevel(logging.ERROR)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  

# Quantization config (change 4-bit or 8-bit depending on what works)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

def simple_hindi_answer(crop, soil, weather, market):
    prompt = f"""आप किसान मित्र हैं।
अनुशंसित फसल: {crop}
मिट्टी: {soil}
मौसम: {weather}
बाज़ार: {market}

कृपया आसान और साधारण हिंदी में, छोटे वाक्यों का उपयोग करके समझाइए।
तकनीकी शब्दों का उपयोग न करें।
बिंदुवार उत्तर दीजिए।
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=150,
        temperature=0.6,
        top_p=0.9,
        do_sample=True
    )
    output = model.generate(**inputs, generation_config=generation_config)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# Test
print(simple_hindi_answer("गेहूँ (Wheat)", "pH 6.5", "बरसात 300mm", "मंडी भाव अच्छा है"))
