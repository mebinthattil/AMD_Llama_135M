from transformers import LlamaForCausalLM, AutoTokenizer
import torch

model_path = "./result"  

model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def chat():
    print("Chat")
    print("Type 'exit' to quit.")
    
    chat_history = "" 
    
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bye")
            break
        
        chat_history += f"You: {user_input}\n"
        prompt = chat_history + "Bot:"
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)

        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9)

        response = tokenizer.decode(tokens[0], skip_special_tokens=True)
        bot_response = response[len(prompt):].strip()
        print(f"Bot: {bot_response}")
        
        chat_history += f"Bot: {bot_response}\n"

if __name__ == "__main__":
    chat()

