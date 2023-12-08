from transformers import AutoModelForCausalLM, AutoTokenizer

print("Libraries imported")

model_name = "mistralai/Mistral-7B-v0.1"
prompt = "Tell me about gravity"
# access_token = "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"


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True,  
                                            #  use_auth_token=access_token
                                             )
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

output = model.generate(**model_inputs)

print(tokenizer.decode(output[0], skip_special_tokens=True))