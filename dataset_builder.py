import json
import os
import pandas as pd
from datasets import load_dataset
from ollama import Client

# Force Python to bypass campus proxies and connect directly to local Ollama
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

print("Downloading MedQuAD Dataset from Hugging Face...")
# Load the actual text dataset
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
df = dataset.to_pandas()

# Filter for keywords relevant to your project
print("Filtering for palliative/hospice care cases...")
palliative_data = df[df['Question'].str.contains('hospice|palliative|terminal|pain', case=False, na=False)].head(5) 

new_json_cases = []
# Create a direct connection to your local AI
local_client = Client(host='http://127.0.0.1:11434')

print("Asking Ollama to format the data...")
for index, row in palliative_data.iterrows():
    prompt = f"""
    Convert this medical Q&A into a single JSON object for a palliative care app.
    Question: {row['Question']}
    Answer: {row['Answer']}
    
    Format it EXACTLY like this (use simple language for caregivers):
    {{
      "case_id": "AUTO-{index}",
      "condition": "Short title of condition",
      "primary_symptom": "The question asked",
      "common_terms": "simple everyday words for this",
      "patient_profile": {{
        "age_group": "Any Age",
        "palliative_stage": "Advanced",
        "vital_concerns": "Main worry"
      }},
      "recommended_interventions": ["Step 1", "Step 2"],
      "contraindications_warnings": ["What NOT to do"],
      "evidence_reference": "MedQuAD Dataset"
    }}
    Return ONLY valid JSON. Do not include markdown blocks.
    """
    
    try:
        # Use the proxy-bypassed client to talk to your model
        response = local_client.chat(model='qwen2.5-coder:7b', messages=[{'role': 'user', 'content': prompt}])
        print(f"Generated case AUTO-{index}")
        
        # Clean the output and append
        clean_json = response['message']['content'].strip('```json').strip('```').strip()
        new_json_cases.append(json.loads(clean_json))
        
    except Exception as e:
        print(f"Failed to process case AUTO-{index}: {e}")

# Save to a new file
with open("expanded_cases.json", "w", encoding="utf-8") as f:
    json.dump(new_json_cases, f, indent=2)

print("Done! Open expanded_cases.json to see your new AI-generated dataset.")