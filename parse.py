import os
from PIL import Image
from PIL.ExifTags import TAGS
import json
import re
import requests
import time
from tqdm import tqdm

# Кеш для API запросов
api_cache = {}

def extract_user_comment(image_path):
    """Extracts UserComment or exif data."""
    try:
        with Image.open(image_path) as img:
            exif_data = img.info.get("exif")
            exif_tags = {TAGS[k]: v for k, v in img._getexif().items() if k in TAGS}
            user_comment = exif_tags.get("UserComment")
            if user_comment:
                return user_comment[8:].decode('utf-16be', errors='ignore').strip() if isinstance(user_comment, bytes) else user_comment.strip()
            elif exif_data:
                return exif_data.decode('utf-8', errors='ignore').strip() if isinstance(exif_data, bytes) else exif_data.strip()
            return None
    except Exception as e:
        print(f"Error extracting data from {image_path}: {e}")
        return None

def get_model_info_from_api_version(version_id, file_pbar):
    """Gets model info from Civitai API, using cache. Updates file_pbar."""
    if version_id in api_cache:
        file_pbar.update(1)
        return api_cache[version_id]

    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'model' in data and 'name' in data['model'] and 'type' in data['model'] and 'files' in data:
            model_info = {
                "type": data['model']['type'],
                "modelName": data['model']['name'],
                "modelVersionName": data['name'],
                "files": data['files']
            }
            api_cache[version_id] = model_info
            file_pbar.update(1)
            return model_info
        else:
            print(f"Necessary model info not found in API response for {url}")
            file_pbar.update(1)
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request error for {url}: {e}")
        file_pbar.update(1)
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error for {url}: {e}")
        file_pbar.update(1)
        return None

def extract_civitai_info_from_urn(urn):
    """Extracts modelVersionId from a Civitai URN."""
    match = re.search(r"civitai:.*?@(\d+)", urn)
    return int(match.group(1)) if match else None

def get_hash_from_files(files, hash_type="AutoV3"):
    """Extracts hash from model files."""
    for file_data in files:
        if "hashes" in file_data and hash_type in file_data["hashes"]:
            return file_data["hashes"][hash_type]
    return None

def count_api_calls(user_comment):
    """Counts the *required* number of Civitai API calls."""
    try:
        data = json.loads(user_comment)
        count = 0
        if "extraMetadata" in data:
            extra_metadata_str = data.get("extraMetadata", "")
            extra_metadata_str = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), extra_metadata_str)
            extra_metadata = json.loads(extra_metadata_str)
            count += len(extra_metadata.get("resources", []))
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    if value.get("class_type") == "CheckpointLoaderSimple" and "inputs" in value and "ckpt_name" in value["inputs"]:
                        if extract_civitai_info_from_urn(value["inputs"]["ckpt_name"]):
                            count += 1
                    elif value.get("class_type") == "LoraLoader" and "inputs" in value and "lora_name" in value["inputs"]:
                        if extract_civitai_info_from_urn(value["inputs"]["lora_name"]):
                            count += 1
            positive_prompt = ""
            negative_prompt = ""

            if "extraMetadata" in data:
                extra_metadata_str = data.get("extraMetadata", "")
                extra_metadata_str = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), extra_metadata_str)
                extra_metadata = json.loads(extra_metadata_str)
                positive_prompt = extra_metadata.get("prompt", "")
                negative_prompt = extra_metadata.get("negativePrompt", "")
            else:
                for key, value in data.items():
                    if isinstance(value, dict):
                        if value.get("_meta", {}).get("title") == "Positive":
                            positive_prompt = value["inputs"]["text"]
                        elif value.get("_meta", {}).get("title") == "Negative":
                            negative_prompt = value["inputs"]["text"]

            for prompt_text in [positive_prompt, negative_prompt]:
                count += len(list(re.finditer(r"embedding:urn:air:sd[xl]?:embedding:civitai:(\d+)@(\d+)", prompt_text)))

        return count
    except json.JSONDecodeError:
        return 0


def parse_and_save_metadata(user_comment, image_path, overall_pbar):
    """Parses metadata, saves to file, updates progress bars."""

    positive_prompt = ""
    negative_prompt = ""
    steps = cfg_scale = sampler_name = seed = None
    civitai_resources = []
    model_name = model_hash = None
    lora_hashes = {}
    lora_info = {}
    parsed_from_extra_metadata = False

    # --- Get the directory of the input image ---
    image_dir = os.path.dirname(image_path)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(image_dir, base_filename + ".txt")


    num_api_calls = count_api_calls(user_comment)
    with tqdm(total=num_api_calls, desc=f"Processing: {base_filename}", leave=False) as file_pbar:
        try:
            data = json.loads(user_comment)
            if "extraMetadata" in data:
                parsed_from_extra_metadata = True
                extra_metadata_str = data.get("extraMetadata", "")
                extra_metadata_str = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), extra_metadata_str)
                try:
                    extra_metadata = json.loads(extra_metadata_str)
                    positive_prompt = extra_metadata.get("prompt", "")
                    negative_prompt = extra_metadata.get("negativePrompt", "")
                    steps = extra_metadata.get("steps")
                    cfg_scale = extra_metadata.get("cfgScale")
                    sampler_name = extra_metadata.get("sampler")
                    seed = extra_metadata.get("seed")
                    positive_prompt = positive_prompt.replace('\n', ' ').strip()
                    negative_prompt = negative_prompt.replace('\n', ' ').strip()
                    resources = extra_metadata.get("resources", [])

                    for resource in resources:
                        version_id = resource.get("modelVersionId")
                        strength = resource.get("strength", 1.0)
                        if version_id:
                            model_info = get_model_info_from_api_version(version_id, file_pbar)
                            if model_info:
                                resource_data = {
                                    "type": model_info["type"],
                                    "modelName": model_info["modelName"],
                                    "modelVersionName": model_info["modelVersionName"],
                                    "modelVersionId": version_id,
                                    "files": model_info["files"],
                                    "strength": strength
                                }
                                if model_info["type"] == "LORA":
                                    for file_data in model_info.get("files", []):
                                        if file_data.get("name", "").endswith(".safetensors"):
                                            filename = file_data["name"][:-len(".safetensors")]
                                            resource_data["filename"] = filename
                                            lora_info[filename] = strength
                                            break
                                civitai_resources.append(resource_data)
                            else:
                                print(f"Could not fetch info for resource with version ID: {version_id} from extraMetadata")
                        else:
                            print(f"modelVersionId not found in resource: {resource}")

                except json.JSONDecodeError as e:
                    print(f"Error decoding extraMetadata JSON: {e}")
                    parsed_from_extra_metadata = False

        except json.JSONDecodeError as e:
            print(f"Error decoding initial JSON from user comment: {e}. Trying fallback parsing.")
            try:
                data = json.loads(user_comment.replace('extraMetadata', ''))
            except json.JSONDecodeError as e2:
                print(f"Error decoding JSON from user comment (fallback): {e2}")
                return

        if not parsed_from_extra_metadata:
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

                    elif value.get("class_type") == "CheckpointLoaderSimple" and "inputs" in value and "ckpt_name" in value["inputs"]:
                        ckpt_name = value["inputs"]["ckpt_name"]
                        version_id = extract_civitai_info_from_urn(ckpt_name)
                        if version_id:
                            model_info = get_model_info_from_api_version(version_id, file_pbar)
                            if model_info:
                                model_name = model_info["modelName"]
                                model_hash = get_hash_from_files(model_info["files"])
                                civitai_resources.append({
                                    "type": model_info["type"],
                                    "modelName": model_info["modelName"],
                                    "modelVersionName": model_info["modelVersionName"],
                                    "modelVersionId": version_id,
                                     "files": model_info["files"]
                                })
                            else:
                                print(f"Could not fetch info for checkpoint with version ID: {version_id}")

                    elif value.get("class_type") == "LoraLoader" and "inputs" in value and "lora_name" in value["inputs"]:
                        lora_name = value["inputs"]["lora_name"]
                        strength_model = value["inputs"].get("strength_model", 1.0)
                        version_id = extract_civitai_info_from_urn(lora_name)
                        if version_id:
                            model_info = get_model_info_from_api_version(version_id, file_pbar)
                            if model_info:
                                lora_filename = next((fd["name"][:-len(".safetensors")] for fd in model_info.get("files", []) if fd.get("name", "").endswith(".safetensors")), None)
                                if lora_filename:
                                    lora_hash = get_hash_from_files(model_info["files"])
                                    if lora_hash:
                                         lora_hashes[lora_filename] = lora_hash
                                    lora_info[lora_filename] = strength_model
                                    civitai_resources.append({
                                        "type": model_info["type"],
                                        "weight": strength_model,
                                        "modelName": model_info["modelName"],
                                        "modelVersionName": model_info["modelVersionName"],
                                        "modelVersionId": version_id,
                                        "filename": lora_filename,
                                        "files": model_info["files"],
                                        "strength": strength_model
                                    })
                            else:
                                print(f"Could not fetch info for Lora with version ID: {version_id}")

            positive_prompt = positive_prompt.replace('\n', ' ').strip()
            negative_prompt = negative_prompt.replace('\n', ' ').strip()
            parts = positive_prompt.split(", ")
            positive_prompt_parts = [part for part in parts if not part.startswith("embedding:urn")]
            positive_prompt = ", ".join(positive_prompt_parts)
            parts = negative_prompt.split(", ")
            negative_prompt_parts = [part for part in parts if not part.startswith("embedding:urn")]
            negative_prompt = ", ".join(negative_prompt_parts)

            for prompt_text in [positive_prompt, negative_prompt]:
                for match in re.finditer(r"embedding:urn:air:sd[xl]?:embedding:civitai:(\d+)@(\d+)", prompt_text):
                    model_id = match.group(1)
                    version_id = int(match.group(2))
                    model_info = get_model_info_from_api_version(version_id, file_pbar)
                    if model_info:
                        civitai_resources.append({
                            "type": res_type,
                            "modelVersionId": version_id,
                            "files": model_info["files"]
                        })
                    else:
                        print(f"Could not fetch model info for embedding with version ID: {version_id}")

        for resource in civitai_resources:
            if resource.get("type") == "LORA" and "filename" in resource:
                lora_hash = get_hash_from_files(resource["files"])
                if lora_hash:
                    lora_hashes[resource["filename"]] = lora_hash

        with Image.open(image_path) as img:
            width, height = img.size
        size_str = f"{width}x{height}"

        lora_strings = [f"<lora:{filename}:{strength}>" for filename, strength in lora_info.items()]
        if lora_strings:
            positive_prompt += ", " + ", ".join(lora_strings)

        positive_prompt = re.sub(r" {2,}", " ", positive_prompt)
        negative_prompt = re.sub(r" {2,}", " ", negative_prompt)

        output_lines = [positive_prompt]
        if negative_prompt:
            output_lines.append(f"Negative prompt: {negative_prompt}")

        params = []
        if steps is not None:      params.append(f"Steps: {steps}")
        if sampler_name is not None:   params.append(f"Sampler: {sampler_name}"); params.append("Schedule type: Automatic")
        if cfg_scale is not None:     params.append(f"CFG scale: {cfg_scale}")
        if seed is not None:          params.append(f"Seed: {seed}")
        params.append(f"Size: {size_str}")

        if model_hash:     params.append(f"Model hash: {model_hash}")
        if model_name:     params.append(f"Model: {model_name}")
        elif civitai_resources:
            for res in civitai_resources:
                if res["type"] == "Checkpoint":
                    model_name = res["modelName"]
                    model_hash = get_hash_from_files(res["files"])
                    if model_hash:
                        params.append(f"Model hash: {model_hash}")
                    params.append(f"Model: {model_name}")
                    break

        if lora_hashes:
            lora_hash_str = ", ".join([f"{lora}: {h}" for lora, h in lora_hashes.items()])
            params.append(f"Lora hashes: \"{lora_hash_str}\"")
        output_lines.append(", ".join(params))

        try:
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.write("\n".join(output_lines))
        except Exception as e:
            print(f"Error saving file {output_file}: {e}")
        finally:
            file_pbar.close()
            overall_pbar.update(1)

def process_images_in_folder(folder_path=None):
    """Processes all images in the folder and subfolders."""
    if folder_path is None:
        folder_path = os.getcwd()

    files_to_process = [(root, file_name) for root, _, files in os.walk(folder_path)
                        for file_name in files if file_name.lower().endswith(('.jpg', '.jpeg'))]

    with tqdm(total=len(files_to_process), desc="Overall Progress") as overall_pbar:
        for root, file_name in files_to_process:
            image_path = os.path.join(root, file_name)
            # base_filename = os.path.splitext(file_name)[0]  <- No longer needed here
            # output_file = os.path.join(root, base_filename + ".txt") <- No longer needed here

            # --- Check for existence using the *correct* path ---
            image_dir = os.path.dirname(image_path)  # Get directory of *image*
            base_filename = os.path.splitext(os.path.basename(image_path))[0] #correct filename
            output_file = os.path.join(image_dir, base_filename + ".txt") # Use image directory

            if os.path.exists(output_file):
                print(f"File {output_file} already exists. Skipping {file_name}.\n")
                overall_pbar.update(1)
                continue

            print(f"Processing file: {image_path}")

            user_comment = extract_user_comment(image_path)
            if not user_comment:
                print(f"UserComment or exif data missing or empty in {file_name}.\n")
                overall_pbar.update(1)
                continue

            parse_and_save_metadata(user_comment, image_path, overall_pbar) # Pass image_path

if __name__ == "__main__":
    process_images_in_folder()
    print("Done.")
