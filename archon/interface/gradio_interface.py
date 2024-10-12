import gradio as gr
import json
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archon import Archon

def load_config_files():
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
    return config_files

def load_config(config_file):
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    with open(os.path.join(config_dir, config_file), 'r') as f:
        return json.load(f)

def generate_response(config_file, prompt):
    config = load_config(config_file)
    archon = Archon(config)
    response = archon.generate([{"role": "user", "content": prompt}])
    return response

def create_interface():
    config_files = load_config_files()

    with gr.Blocks(title="Archon GPT Interface") as interface:
        gr.Markdown("# Archon GPT Interface")
        
        with gr.Row():
            config_dropdown = gr.Dropdown(choices=config_files, label="Select Config File")
        
        with gr.Row():
            input_text = gr.Textbox(lines=5, label="Input Prompt")
        
        with gr.Row():
            submit_button = gr.Button("Generate Response")
        
        with gr.Row():
            output_text = gr.Textbox(lines=10, label="Generated Response")

        submit_button.click(
            generate_response,
            inputs=[config_dropdown, input_text],
            outputs=[output_text]
        )

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()

