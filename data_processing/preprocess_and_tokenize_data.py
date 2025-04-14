import pandas as pd
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset
import os

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
MODEL_NAME = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class PRDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
    
    def __len__(self):
        return len(self.inputs['input_ids'])
    
    def __getitem__(self, index):
        return {
            'input_ids' : torch.tensor(self.inputs['input_ids'][index]),
            'attention_mask' : torch.tensor(self.inputs['attention_mask'][index]),
            'labels' : torch.tensor(self.outputs['input_ids'][index])
        }
    
def tokenize_data(df):
    input_encodings = tokenizer(
        df["input"].tolist(),
        truncation = True,
        padding = 'max_length',
        max_length = MAX_INPUT_LENGTH,
        return_tensors = "pt"
    )

    output_encodings = tokenizer(
        df["output"].tolist(),
        truncation = True,
        padding = 'max_length',
        max_length = MAX_OUTPUT_LENGTH,
        return_tensors = "pt"
    )

    return PRDataset(input_encodings, output_encodings)
    
def main():
    base_path = "/Volumes/Data/WorkSpace/pr-review-ai/formatted_data"
    save_path = "/Volumes/Data/WorkSpace/pr-review-ai/processed_data"

    os.makedirs(save_path, exist_ok=True)

    for split in ["formatted_review_data_test", "formatted_review_data_train", "formatted_review_data_val"]:
        print(f"Processing {split}.csv")

        df = pd.read_csv(f"{base_path}/{split}.csv")

        dataset = tokenize_data(df)

        torch.save(dataset, f"{save_path}/{split}_dataset.pt")
        print(f"Saved {split} dataset to {save_path}/{split}_dataset.pt")

if __name__ == "__main__":
    main()