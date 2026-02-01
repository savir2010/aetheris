import torch    
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

# 1. Setup for M3 Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Use float16 for M3 performance
torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

MODEL_ID = "microsoft/Florence-2-base-ft"

print(f"Loading {MODEL_ID} on {device}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# 2. Load image and define task/target
image_path = "v2.png"
image = Image.open(image_path).convert("RGB")
w, h = image.width, image.height

TARGET_TEXT = "IAM"
task_prompt = "<OCR_WITH_REGION>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
if device == "mps": inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        num_beams=3
    )

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
results = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))

# 3. Filter results for TARGET_TEXT
ocr_data = results.get(task_prompt, {})
found = False

for box, label in zip(ocr_data.get("quad_boxes", []), ocr_data.get("labels", [])):
    if TARGET_TEXT.lower() in (label or "").lower():
        # quad_boxes are [x1, y1, x2, y2, x3, y3, x4, y4]
        # Calculate center from the quad points
        cx = int(sum(box[0::2]) / 4)
        cy = int(sum(box[1::2]) / 4)
        
        print(f"MATCH FOUND: '{label}' at [{cx}, {cy}]")
        found = True
        break

if not found:
    print(f"'{TARGET_TEXT}' not detected in OCR. Trying phrase grounding fallback...")