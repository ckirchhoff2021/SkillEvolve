import os
import yaml
from dotenv import load_dotenv
from tools.logger import setup_logger


setup_logger("skillevolve.chat", log_file="logs/hotpotqa.log", stream=False)
setup_logger("skillevolve.responses", log_file="logs/hotpotqa.log", stream=False)

from engine.trainer import SkillTrainer
from models import ResponsesModel, ChatModel
from rollouts.hotpotqa import HotpotQARollout

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
    
    dataset_config = yaml.safe_load(open("datas/hotpotqa.yaml", 'r'))
    rollout = HotpotQARollout(target_model, dataset_config)
    
    rollout.rollout_train("Test")

