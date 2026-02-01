import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from PIL import Image
import os

MODEL_ID = "microsoft/Florence-2-large"
IMAGE_PATH = "v2.png"
TARGET_TEXT = "iam"

# Setup for M3
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 

print(f"Loading {MODEL_ID}...")
model = Florence2ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def get_coordinates(image_path, target_text):
    if not os.path.exists(image_path):
        return f"Error: {image_path} not found."

    raw_img = Image.open(image_path).convert("RGB")
    
    # Using Phrase Grounding to find specific text
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = f"{task_prompt}{target_text}"
    
    inputs = processor(text=prompt, images=raw_img, return_tensors="pt")
    inputs = {k: v.to(device).to(torch_dtype) if v.dtype == torch.float32 else v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3
        )

    prediction = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        prediction, 
        task=task_prompt, 
        image_size=raw_img.size
    )

    # --- Clean Parsing ---
    data = result.get(task_prompt, {})
    if not data or "bboxes" not in data:
        return None

    # Find the best match
    for box, label in zip(data["bboxes"], data["labels"]):
        if target_text.lower() in label.lower():
            x1, y1, x2, y2 = box
            center_x = round((x1 + x2) / 2)
            center_y = round((y1 + y2) / 2)
            return {"center": (center_x, center_y), "box": box, "label": label}
    
    return None

# Execution
print(f"Scanning for '{TARGET_TEXT}'...")
coords = get_coordinates(IMAGE_PATH, TARGET_TEXT)

if coords:
    print("-" * 30)
    print(f"TARGET FOUND: {coords['label']}")
    print(f"CENTER POINT: {coords['center']}")
    print(f"BOUNDING BOX: {coords['box']}")
    print("-" * 30)
else:
    print(f"Could not find '{TARGET_TEXT}' in the image.")