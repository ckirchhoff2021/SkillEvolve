import os
import yaml
from dotenv import load_dotenv
from tools.logger import setup_logger

# 必须在导入 models 之前配置 logger，否则 responses.py 会先创建不带 log_file 的 logger
setup_logger("skillevolve.responses", log_file="logs/text_infer.log")

from engine.trainer import SkillTrainer
from models import ResponsesModel, ChatModel
from rollouts.docvqa import DocVQARollout

load_dotenv()


if __name__ == '__main__':  
    
    local_api_key = "EMPTY"
    local_api_base = "http://[2605:340:cd51:4900:6f7c:1e8d:18b7:ab40]:9001/v1"   #"http://localhost:8000/v1"
    local_model = "Qwen3.5-27B" #"Qwen3.5-4B"

    # target_model = ChatModel("http://localhost:8000/v1", "Empty", "Qwen3.5-0.8B")
    target_model = ChatModel(local_api_base, local_api_key, local_model)
    
    base_url = os.getenv("API_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")
    optimizer_model = ResponsesModel(base_url, api_key, model_name)
    
    dataset_config = yaml.safe_load(open("datas/docvqa.yaml"))
    rollout = DocVQARollout(target_model, dataset_config)
    
    init_skill = dataset_config["dataset_info"]["init_skill"]
    trainer = SkillTrainer(target_model, optimizer_model, rollout, initial_skill_file=init_skill, output_dir="logs")
    trainer.train()

