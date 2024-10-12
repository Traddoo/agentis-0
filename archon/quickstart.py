from archon import Archon
import gradio as gr

# make sure to set your OPENAI_API_KEY environment variable

# Initialize Archon
single_gpt_config = {
    "name": "gpt-4o-single",
    "layers": [
        [
            {
                "type": "generator",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "top_k": 1,
                "temperature": 0.7,
                "max_tokens": 2048,
                "samples": 1,
            }
        ]
    ],
}


archon_gpt_config = {
    "name": "archon-gpt-multi-model",
    "layers": [
        [
            {
                "type": "generator",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "top_k": 1,
                "temperature": 0.7,
                "max_tokens": 2048,
                "samples": 10,
            }
        ],
        [
            {
                "type": "ranker",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 2048,
            }
        ],
        [
            {
                "type": "fuser",
                "model": "gpt-4o",
                "model_type": "OpenAI_API",
                "temperature": 0.7,
                "max_tokens": 2048,
                "samples": 1,
            }
        ],
    ],
}

#################################################

single_gpt = Archon(single_gpt_config)
archon_gpt = Archon(archon_gpt_config)

def generate_responses(user_input):
    instruction = [{"role": "user", "content": user_input}]
    
    single_gpt_response = single_gpt.generate(instruction)
    archon_gpt_response = archon_gpt.generate(instruction)
    
    # Print responses to terminal
    print(f"Single GPT ({single_gpt.config['name']}) Query: {user_input}")
    print(f"Single GPT Response: {single_gpt_response}")
    print("---------------------")
    print(f"Archon GPT ({archon_gpt.config['name']}) Query: {user_input}")
    print(f"Archon GPT Response: {archon_gpt_response}")
    print("=====================")
    
    return f"Single GPT: {single_gpt_response}", f"Archon GPT: {archon_gpt_response}"

# Create Gradio interface
iface = gr.Interface(
    fn=generate_responses,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs=[
        gr.Textbox(label="Single GPT Response"),
        gr.Textbox(label="Archon GPT Response")
    ],
    title="Archon GPT Comparison",
    description="Compare responses from Single GPT and Archon GPT"
)

# Launch the interface
iface.launch()
