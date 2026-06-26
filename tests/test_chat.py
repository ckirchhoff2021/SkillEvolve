from dotenv import load_dotenv
from models import ChatModel
import os


load_dotenv()


def case_001():
    skill_prompt = "请详细描述一下这张图像。"
    image_input = "assets/images/ghibli_converted_portrait.jpg"
    base_url = "http://localhost:8000/v1"
    chat_model = ChatModel(base_url, "Empty", "Qwen3.5-0.8B")
    result = chat_model.infer_with_image(image_input, skill_prompt)
    print(result)


if __name__ == '__main__':
    case_001()