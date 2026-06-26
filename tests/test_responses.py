from dotenv import load_dotenv
from models import ResponsesModel
import os


load_dotenv()

base_url = os.getenv("API_URL")
api_key = os.getenv("API_KEY")
model_name = os.getenv("MODEL_NAME")
chat_model = ResponsesModel(base_url, api_key, model_name)


def case_001():
    skill_prompt = "请详细描述一下这张图像。"
    image_input = "assets/images/ghibli_converted_portrait.jpg"
    result = chat_model.infer_with_image(image_input, skill_prompt)
    print(result)


def case_002():
    skill_prompt = "细节要清晰，人物要丰满。"
    text_input = "请给我讲个故事"
    result = chat_model.infer_with_text(text_input, skill_prompt)
    print(result)


if __name__ == '__main__':
    # case_001()
    case_002()