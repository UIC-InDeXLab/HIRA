from transformers import AutoConfig

# model_id = "Qwen/Qwen2.5-7B"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
config = AutoConfig.from_pretrained(model_id)

print(config)

# Or explicitly inspect fields
for k in vars(config):
    if "pos" in k.lower() or "seq" in k.lower() or "context" in k.lower():
        print(k, "=", getattr(config, k))
