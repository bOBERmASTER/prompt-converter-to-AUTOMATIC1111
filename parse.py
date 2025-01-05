import os
from PIL import Image
from PIL.ExifTags import TAGS
import json
import re
import requests
import time

def extract_user_comment(image_path):
    """Извлекает данные UserComment или exif из метаданных изображения."""
    try:
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif")
            exif_tags = {TAGS[k]: v for k, v in img._getexif().items() if k in TAGS}
            user_comment = exif_tags.get("UserComment")
            if user_comment:
                if isinstance(user_comment, bytes):
                    user_comment = user_comment[8:].decode('utf-16be', errors='ignore')
                    # print(user_comment)
                return user_comment.strip()
            elif exif_data:
                if isinstance(exif_data, bytes):
                    exif_data = exif_data.decode('utf-8', errors='ignore')
                return exif_data.strip()
            return None
    except Exception as e:
        print(f"Ошибка при извлечении данных из {image_path}: {e}")
        return None

def get_model_info_from_api_version(version_id):
    """
    Получает информацию о модели с Civitai API по ID версии.

    Args:
        version_id: ID версии модели на Civitai.

    Returns:
        Словарь с информацией о модели (type, modelName, modelVersionName, files) или None, если не удалось получить.
    """
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        if 'model' in data and 'name' in data['model'] and 'type' in data['model'] and 'files' in data:
            return {
                "type": data['model']['type'],
                "modelName": data['model']['name'],
                "modelVersionName": data['name'],
                "files": data['files']
            }
        else:
            print(f"Не найдена необходимая информация о модели в ответе API {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Ошибка при разборе JSON от API {url}: {e}")
        return None

def extract_civitai_info_from_urn(urn):
    """Extracts modelVersionId from a Civitai URN."""
    match = re.search(r"civitai:.*?@(\d+)", urn)
    if match:
        return int(match.group(1))
    return None

def parse_and_save_metadata(user_comment, base_filename):
    """
    Parses the user comment for prompt information and Civitai resources,
    and saves it to a .txt file.
    """
    try:
        data = json.loads(user_comment.replace('extraMetadata', ''))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from user comment: {e}")
        return

    positive_prompt = ""
    negative_prompt = ""
    steps = None
    cfg_scale = None
    sampler_name = None
    seed = None
    extra_data = {}
    civitai_resources = []

    for key, value in data.items():
        if isinstance(value, dict):
            if value.get("_meta", {}).get("title") == "Positive":
                positive_prompt = value["inputs"]["text"]
            elif value.get("_meta", {}).get("title") == "Negative":
                negative_prompt = value["inputs"]["text"]
            elif value.get("_meta", {}).get("title") == "KSampler":
                if "inputs" in value:
                    steps = value["inputs"].get("steps")
                    cfg_scale = value["inputs"].get("cfg")
                    sampler_name = value["inputs"].get("sampler_name")
                    seed = value["inputs"].get("seed")
                ksampler_meta = value["_meta"].copy()
                if "title" in ksampler_meta:
                    del ksampler_meta["title"]
                extra_data.update(ksampler_meta)
            elif value.get("class_type") == "CheckpointLoaderSimple":
                if "inputs" in value and "ckpt_name" in value["inputs"]:
                    ckpt_name = value["inputs"]["ckpt_name"]
                    version_id = extract_civitai_info_from_urn(ckpt_name)
                    if version_id:
                        model_info = get_model_info_from_api_version(version_id)
                        if model_info:
                            civitai_resources.append({
                                "type": model_info["type"],
                                "modelName": model_info["modelName"],
                                "modelVersionName": model_info["modelVersionName"],
                                "modelVersionId": version_id
                            })
                            print(f"Added checkpoint to civitai_resources: {civitai_resources[-1]}")
                            time.sleep(0.5) # Don't overload the API
                        else:
                            print(f"Could not fetch info for checkpoint with version ID: {version_id}")
            elif value.get("class_type") == "LoraLoader":
                if "inputs" in value and "lora_name" in value["inputs"]:
                    lora_name = value["inputs"]["lora_name"]
                    strength_model = value["inputs"].get("strength_model", 1.0)
                    version_id = extract_civitai_info_from_urn(lora_name)
                    if version_id:
                        model_info = get_model_info_from_api_version(version_id)
                        if model_info:
                            lora_filename = None
                            for file_data in model_info.get("files", []):
                                if file_data.get("name", "").endswith(".safetensors"):
                                    lora_filename = file_data["name"][:-len(".safetensors")]
                                    break
                            if lora_filename:
                                civitai_resources.append({
                                    "type": model_info["type"],
                                    "weight": strength_model,
                                    "modelName": model_info["modelName"],
                                    "modelVersionName": model_info["modelVersionName"],
                                    "modelVersionId": version_id,
                                    "filename": lora_filename
                                })
                                print(f"Added Lora to civitai_resources: {civitai_resources[-1]}")
                                time.sleep(0.5) # Don't overload the API
                            else:
                                print(f"Could not find .safetensors file for Lora with version ID: {version_id}")
                        else:
                            print(f"Could not fetch info for Lora with version ID: {version_id}")

    # Build Lora string for positive prompt
    lora_strings = []
    for resource in civitai_resources:
        if resource.get("type") == "LORA" and "filename" in resource and "weight" in resource:
            lora_strings.append(f"<lora:{resource['filename']}:{resource['weight']}>")

    # Prepend Lora strings to positive prompt
    if lora_strings:
        positive_prompt = ", ".join(lora_strings) + ", " + positive_prompt
        positive_prompt = positive_prompt.strip(", ") # Remove trailing comma if any

    # Удаляем embedding:urn... подстроки (более общий подход)
    parts = positive_prompt.split(", ")
    positive_prompt_parts = [part for part in parts if not part.startswith("embedding:urn")]
    positive_prompt = ", ".join(positive_prompt_parts).replace('\n', ' ')

    parts = negative_prompt.split(", ")
    negative_prompt_parts = [part for part in parts if not part.startswith("embedding:urn")]
    negative_prompt = ", ".join(negative_prompt_parts).replace('\n', ' ')

    # Extract embedding info from positive and negative prompts (keeping the old logic for now)
    for prompt_text, res_type in [(positive_prompt, "embed"), (negative_prompt, "embed")]:
        for match in re.finditer(r"embedding:urn:air:sd[xl]?:embedding:civitai:(\d+)@(\d+)", prompt_text):
            model_id = match.group(1)
            version_id = int(match.group(2))
            civitai_resources.append({
                "type": res_type,
                "modelVersionId": version_id
            })
            print(f"Added Embedding from prompt: {civitai_resources[-1]}")

    other_info = []
    if steps is not None:
        other_info.append(f"Steps: {steps}")
    if sampler_name is not None:
        other_info.append(f"Sampler: {sampler_name}")
    if cfg_scale is not None:
        other_info.append(f"CFG scale: {cfg_scale}")
    if seed is not None:
        other_info.append(f"Seed: {seed}")
    if civitai_resources:
        other_info.append(f"Civitai resources: {json.dumps(civitai_resources)}")
        print(f"Civitai resources being outputted: {json.dumps(civitai_resources)}")
    else:
        print("Civitai resources is empty.")
    if extra_data:
        other_info.append(f"Extra data: {json.dumps(extra_data)}")

    other_info_str = ", ".join(other_info)

    output_file = base_filename + ".txt"
    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write(positive_prompt + "\n")
            outfile.write(f"Negative prompt: {negative_prompt}" + "\n")
            outfile.write(other_info_str)
        print(f"Результаты сохранены в {output_file}")
    except Exception as e:
        print(f"Ошибка при сохранении файла {output_file}: {e}")

def process_images_in_folder():
    """Обрабатывает все изображения в папке."""
    current_folder = os.getcwd()
    for file_name in os.listdir(current_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg')):
            image_path = os.path.join(current_folder, file_name)
            base_filename = os.path.splitext(file_name)[0]
            output_file = base_filename + ".txt"

            if os.path.exists(output_file):
                print(f"Файл {output_file} уже существует. Пропускаем {file_name}")
                continue

            print(f"Обработка файла: {file_name}")

            user_comment = extract_user_comment(image_path)
            if not user_comment:
                print(f"Поле UserComment или exif отсутствует или пустое в {file_name}")
                continue

            if "resource-stack" in user_comment:
                print(f"Обнаружены 'resource-stack' в {file_name}. Начинаю обработку.")
                parse_and_save_metadata(user_comment, base_filename)
            else:
                print(f"Не найдено 'resource-stack' в {file_name}. Файл не будет обработан.")

if __name__ == "__main__":
    process_images_in_folder()
    print("Завершено.")
