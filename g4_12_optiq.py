from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-4-12B-it-OptiQ-4bit")
response = generate(
    model, tokenizer,
    prompt="Explain quantum information theory and why it is important to understand in simple terms.",
    max_tokens=200,
)
