# fast and easy converter civitai (??) prompt to AUTOMATIC1111

prompt what do you may convert is json with `resource-stack` at the beginning

have AIR's (civitai URN) `urn:air:sd1:checkpoint:civitai:4384@128713`

- [example of image ready for convert](https://civitai.com/images/21793336) 
- [another example of image](https://civitai.com/images/32155314)

for run script you need:
1. python3
2. PIL lib `pip install pillow`
3. put script in folder with images
4. run
5. .txt files will contain prompt you need

## how it works
1. Metadata Extraction: It reads the metadata of JPEG images, specifically looking for a "UserComment" or EXIF data field. This field is expected to contain information about how the image was generated.
2. JSON Parsing: If a "UserComment" is found and contains a "resource-stack", the script attempts to parse it as JSON. This JSON is assumed to hold details about the generation process.
3. Information Retrieval: From the parsed JSON, it identifies and extracts key information:
   - Prompts: Positive and negative prompts used for generation.
   - Sampler Settings: Parameters like steps, CFG scale, sampler name, and seed.
   - Civitai Resources: Information about models, LoRAs, and embeddings used, identified by their Civitai URNs.
4. Civitai API Interaction: For each identified Civitai resource (checkpoint or Lora), the script uses the Civitai API to fetch more detailed information about the model based on its version ID. This includes the model's type, name, version name, and associated files.
5. Lora Formatting: For LoRAs, it constructs a specific string format (<lora:filename:weight>) using the model's filename and strength/weight, to be added to the positive prompt.
6. Output Generation: The extracted information, including the formatted Lora strings (added to the positive prompt), the negative prompt, and other generation settings, is formatted and written into a .txt file. This file is created in the same directory as the original image and has the same name.
7. Automation: The script is designed to process all JPEG images within the directory it's run from.

original prompt
```json
{"resource-stack":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"urn:air:sd1:checkpoint:civitai:4384@128713"}},"resource-stack-1":{"class_type":"LoraLoader","inputs":{"lora_name":"urn:air:sd1:lora:civitai:64560@69190","strength_model":0.45,"strength_clip":1,"model":["resource-stack",0],"clip":["resource-stack",1]}},"resource-stack-2":{"class_type":"LoraLoader","inputs":{"lora_name":"urn:air:sd1:lora:civitai:150182@167845","strength_model":0.8,"strength_clip":1,"model":["resource-stack-1",0],"clip":["resource-stack-1",1]}},"resource-stack-3":{"class_type":"LoraLoader","inputs":{"lora_name":"urn:air:sd1:lora:civitai:342493@383406","strength_model":0.6,"strength_clip":1,"model":["resource-stack-2",0],"clip":["resource-stack-2",1]}},"resource-stack-4":{"class_type":"LoraLoader","inputs":{"lora_name":"urn:air:sd1:lora:civitai:24934@30200","strength_model":1,"strength_clip":1,"model":["resource-stack-3",0],"clip":["resource-stack-3",1]}},"6":{"class_type":"smZ CLIPTextEncode","inputs":{"text":"Chinese horror, maffia,  score_9, score_8_up, score_7_up,  solo, playing banjo, necktie, official suit,\n","parser":"A1111","text_g":"","text_l":"","ascore":2.5,"width":0,"height":0,"crop_w":0,"crop_h":0,"target_width":0,"target_height":0,"smZ_steps":1,"mean_normalization":true,"multi_conditioning":true,"use_old_emphasis_implementation":false,"with_SDXL":false,"clip":["resource-stack-4",1]},"_meta":{"title":"Positive"}},"7":{"class_type":"smZ CLIPTextEncode","inputs":{"text":"embedding:urn:air:sd1:embedding:civitai:99890@106916, score_4, score_5, score_6, score_1, score_2, score_3, noise, lowres, low quality, bad anatomy, sign, score_1, score_2, score_3, noise, lowres, low quality, bad anatomy, sign, watermark,\nSteps: 38,  Sampler: Euler a, CFG scale: 7, Clip skip: 2","parser":"A1111","text_g":"","text_l":"","ascore":2.5,"width":0,"height":0,"crop_w":0,"crop_h":0,"target_width":0,"target_height":0,"smZ_steps":1,"mean_normalization":true,"multi_conditioning":true,"use_old_emphasis_implementation":false,"with_SDXL":false,"clip":["resource-stack-4",1]},"_meta":{"title":"Negative"}},"17":{"class_type":"LoadImage","inputs":{"image":"https://orchestration.civitai.com/v2/consumer/blobs/BJF76XQDRD33DH453VWRQBJSJ0","upload":"image"},"_meta":{"title":"Image Load"}},"18":{"class_type":"VAEEncode","inputs":{"pixels":["17",0],"vae":["resource-stack",2]},"_meta":{"title":"VAE Encode"}},"11":{"class_type":"KSampler","inputs":{"sampler_name":"dpmpp_2m","scheduler":"karras","seed":2433638737,"steps":31,"cfg":3,"denoise":0.55,"model":["resource-stack-4",0],"positive":["6",0],"negative":["7",0],"latent_image":["18",0]},"_meta":{"title":"KSampler"}},"13":{"class_type":"VAEDecode","inputs":{"samples":["11",0],"vae":["resource-stack",2]},"_meta":{"title":"VAE Decode"}},"12":{"class_type":"SaveImage","inputs":{"filename_prefix":"ComfyUI","images":["13",0]},"_meta":{"title":"Save Image"}},"extra":{"airs":["urn:air:sd1:checkpoint:civitai:4384@128713","urn:air:sd1:lora:civitai:64560@69190","urn:air:sd1:lora:civitai:150182@167845","urn:air:sd1:lora:civitai:342493@383406","urn:air:sd1:lora:civitai:24934@30200","urn:air:sd1:embedding:civitai:99890@106916"]}}	
```

converted prompt. 
  - 3 string
  - all data gets from civitai api
  - if there are loras are added to the beginning of the positive prompt
```csv
<lora:watercolorV1:0.45>, <lora:Horror:0.8>, <lora:Chinese horror_v1:0.6>, <lora:fantasyV1.1:1>, Chinese horror, maffia,  score_9, score_8_up, score_7_up,  solo, playing banjo, necktie, official suit, 
Negative prompt: score_4, score_5, score_6, score_1, score_2, score_3, noise, lowres, low quality, bad anatomy, sign, score_1, score_2, score_3, noise, lowres, low quality, bad anatomy, sign, watermark, Steps: 38,  Sampler: Euler a, CFG scale: 7, Clip skip: 2
Steps: 31, Sampler: dpmpp_2m, CFG scale: 3, Seed: 2433638737, Civitai resources: [{"type": "Checkpoint", "modelName": "DreamShaper", "modelVersionName": "8", "modelVersionId": 128713}, {"type": "LORA", "weight": 0.45, "modelName": "WATERCOLOR", "modelVersionName": "v1.0", "modelVersionId": 69190, "filename": "watercolorV1"}, {"type": "LORA", "weight": 0.8, "modelName": "Horror & Creepy", "modelVersionName": "v1.0", "modelVersionId": 167845, "filename": "Horror"}, {"type": "LORA", "weight": 0.6, "modelName": "Chinese Horror", "modelVersionName": "Chinese horror-v1.0", "modelVersionId": 383406, "filename": "Chinese horror_v1"}, {"type": "LORA", "weight": 1, "modelName": "Dark Fantasy", "modelVersionName": "v1.1", "modelVersionId": 30200, "filename": "fantasyV1.1"}]
```
