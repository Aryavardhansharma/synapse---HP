from huggingface_hub import login
login("hugging_face")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Use the chat-optimized 7B model
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 8-bit is safer in free Colab (4-bit may fail if GPU doesn’t support it)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,   # More stable on Colab Free
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"]
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Test prompt
prompt = "Explain what reinforcement learning is in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
