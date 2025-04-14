import pandas as pd

raw_review_df = pd.read_csv("/Volumes/Data/WorkSpace/pr-review-ai/raw_data/test.csv")

training_pairs = []
for _, review_record in raw_review_df.iterrows():
    review_context = (
        f"Reviewer: {review_record['Reviewer']} | "
        f"Author: {review_record['Author']} | "
        f"File: {review_record['File']} | "
        f"Category: {review_record['Category']} | "
        f"CodeDiff: {review_record['CodeDiff']}"
    )

    review_comment = review_record['Comment']

    training_pairs.append({
        "input" : review_context,
        "output" : review_comment
    })

formatted_training_df = pd.DataFrame(training_pairs)
formatted_training_df.to_csv("formatted_review_data_val.csv", index=False)

print("✅ Transformer-ready dataset saved as 'formatted_review_test.csv'")
