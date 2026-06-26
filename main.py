import os
from engine.trainer import SkillTrainer
from models import ResponsesModel, ChatModel
from rollouts.docvqa import DocVQARollout

from dotenv import load_dotenv

load_dotenv()


if __name__ == '__main__':  
    target_model = ChatModel("http://localhost:8000/v1", "Empty", "Qwen3.5-0.8B")
    
    base_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")
    optimizer_model = ResponsesModel(base_url, api_key, model_name)
    
    output_dir = "outputs"    
    rollout = DocVQARollout(target_model)
    trainer = SkillTrainer(target_model, optimizer_model, rollout, initial_skill_file="datas/docvqa/init_skill.md", output_dir=output_dir)
    
    trainer.train()
