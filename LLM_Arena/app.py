import gradio as gr
from LLM import get_random_models, models, get_response_model
from dotenv import load_dotenv
import os
import pandas as pd
import json

load_dotenv()  # Load environment variables from .env file

# Securely load API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

current_model_indices = get_random_models()
SCORE_FILE = "scores.json"

# Load scores from file or initialize if file doesn't exist
if os.path.exists(SCORE_FILE):
    with open(SCORE_FILE, 'r') as file:
        scores = json.load(file)
else:
    scores = {i: 0 for i in models.keys()}

def save_scores():
    with open(SCORE_FILE, 'w') as file:
        json.dump(scores, file)
save_scores()
def generate_responses(prompt, history1=None, history2=None):
    global current_model_indices
    model1, model2 = current_model_indices[0], current_model_indices[1]
    response1 = get_response_model(model1, prompt)
    response2 = get_response_model(model2, prompt)
    formatted_response1 = [(prompt, response1)]
    formatted_response2 = [(prompt, response2)]
    return formatted_response1, formatted_response2, gr.State(response1), gr.State(response2)

def vote(vote_for_model1, vote_for_model2):
    global scores, current_model_indices
    model1, model2 = current_model_indices[0], current_model_indices[1]
    if vote_for_model1 and vote_for_model2:
        scores[model1] += 1
        scores[model2] += 1
    elif vote_for_model1:
        scores[model1] += 1
    else:
        scores[model2] += 1
    save_scores()
    df = pd.DataFrame(list(scores.items()), columns=["models", "score"])
    return df, model1, model2

def new_round():
    global current_model_indices
    current_model_indices = get_random_models()
    return None, None

def clear_responses():
    return "", "", ""

df = pd.DataFrame(list(scores.items()), columns=["models", "score"])

def main():
    with gr.Blocks(css=".gradio-container {background-color: #282828; color: #FFFFFF;}") as demo:
        with gr.Tab("⚔️Arena"):
            gr.Markdown("## LLM Arena")
            gr.Markdown("📜 Rules")
            rules = """
            - Ask any question to two anonymous models.
            - Vote for your preferred model's response and voting should be fair.
            - Use 'New Round' to start a new set of models.
            - Use 'Clear' to reset inputs and prompt.
            - Model names will be visible only after you vote.
            """
            gr.Markdown(rules)
            gr.Markdown("##👇 Chat now!")
            with gr.Row():
                with gr.Column():
                    response1 = gr.Chatbot(label="Model 1 Response")
                with gr.Column():
                    response2 = gr.Chatbot(label="Model 2 Response")

            with gr.Row():
                prompt = gr.Textbox(label="Enter your prompt:", placeholder="Enter your prompt here...", lines=1)
            with gr.Accordion("🥷 Current Models", open=False):
                with gr.Row():
                    box1 = gr.Textbox(interactive=False)
                    box2 = gr.Textbox(interactive=False)
            gr.Markdown("## 👆 Vote")
            with gr.Row():
                with gr.Column():
                    vote1_btn = gr.Button("Vote for Model 1")
                with gr.Column():
                    vote2_btn = gr.Button("Vote for Model 2")
                with gr.Column():
                    vote3_btn = gr.Button("Tie")
            with gr.Row():
                with gr.Column():
                    new_round_btn = gr.Button("🎲 New Round")
                with gr.Column():
                    clear_btn = gr.Button("Clear")
            
        with gr.Tab("🏆Leaderboard"):
            gr.Markdown("## Models and Scores")
            gr.Markdown("View Leaderboard for different models")
            df = pd.DataFrame(list(scores.items()), columns=["models", "score"])
            table = gr.DataFrame(value=df, interactive=False)

        # Button actions
        prompt.submit(generate_responses, [prompt, gr.State([]), gr.State([])], [response1, response2, gr.State([]), gr.State([])])
        vote1_btn.click(vote, inputs=[gr.State(True), gr.State(False)], outputs=[table, box1, box2])
        vote2_btn.click(vote, inputs=[gr.State(False), gr.State(True)], outputs=[table, box1, box2])
        vote3_btn.click(vote, inputs=[gr.State(True), gr.State(True)], outputs=[table, box1, box2])
        new_round_btn.click(new_round, outputs=[response1, response2])
        clear_btn.click(clear_responses, outputs=[response1, response2, prompt])

    demo.launch()

if __name__ == "__main__":
    main()
