import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.preprocess_and_tokenize_data import PRDataset

model_path = "/Volumes/Data/WorkSpace/pr-review-ai/Trained_t5_MODEL"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()

torch.serialization.add_safe_globals([PRDataset])
test_dataset = torch.load("/Volumes/Data/WorkSpace/pr-review-ai/processed_data/formatted_review_data_test_dataset.pt",
                          weights_only = False
                          )
test_loader = DataLoader(test_dataset, batch_size = 5)

all_predictions = []
all_labels = []

for batch in tqdm(test_loader):
    with torch.no_grad():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length = 128,
            num_beams = 4
        )

        decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)

        all_predictions.extend(decoded_predictions)
        all_labels.extend(decoded_labels)

df = pd.DataFrame({
    "Ground_Label" : all_labels,
    "Predicted_Labels" : all_predictions
})

df.to_csv("Inference_Results.csv", index = False)

print("Testing on the model is completed and Inference vs Ground Truth file saved successfully.")