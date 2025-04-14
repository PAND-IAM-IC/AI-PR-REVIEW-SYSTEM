from transformers import TrainingArguments
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import random_split
import torch.serialization
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.preprocess_and_tokenize_data import PRDataset

DATA_PATH = "/Volumes/Data/WorkSpace/pr-review-ai/processed_data"
MODEL_NAME = "t5-small"
SAVE_PATH = "/Volumes/Data/WorkSpace/pr-review-ai/Trained_t5_MODEL"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

torch.serialization.add_safe_globals([PRDataset])

training_dataset = torch.load(f"{DATA_PATH}/formatted_review_data_train_dataset.pt", weights_only = False)
validation_dataset = torch.load(f"{DATA_PATH}/formatted_review_data_val_dataset.pt", weights_only = False)

training_args = TrainingArguments(
    output_dir = SAVE_PATH,
    evaluation_strategy = "epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    num_train_epochs = 13,
    weight_decay = 0.01,
    save_total_limit = 2,
    logging_dir = "./logs",
    logging_steps = 10,
    save_strategy = "epoch",
    load_best_model_at_end = True,
    report_to = "none"
)

def compute_metrics(eval_pred):
    return {}

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = training_dataset,
    eval_dataset = validation_dataset,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Model saved successfully at {SAVE_PATH}")