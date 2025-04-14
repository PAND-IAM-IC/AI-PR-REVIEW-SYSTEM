# PR Review AI ✨

### 🚀 One Line Summary
A fully custom, fine-tuned ML model that reviews pull requests by learning personalized code review habits of team members—no OpenAI APIs involved.

---

## 🔧 What’s Our Hack?

PR Review AI is **not just another GPT wrapper**. It’s a **ground-up, fine-tuned T5 model** that intelligently analyzes pull requests and mimics personalized review behavior—something no generic API can offer.

In our team, we have a dedicated code review group where **tens of PRs are posted daily**, often with repetitive review cycles: 
- Dev pushes code → 
- Reviewer leaves comments → 
- Dev fixes it → 
- PR goes back in the group.

It becomes both tedious and inconsistent. PR Review AI solves this by automating intelligent suggestions based on actual review habits.

For example:
- **Shivam** typically flags *naming convention issues and code repetition*.
- **Pradeep Sir** looks for *redundant code, architecture decisions, and performance*.
- **Shaurya** often forgets to use `StringUtils` for comparisons.

Our model not only flags potential issues—it simulates:
> “Would Pradeep have left a comment here? If yes, what would it be?”

That’s what sets us apart.

---

## 🔍 Why This Matters

Most hackathon projects these days involve calling OpenAI APIs, sending some prompt, and displaying the output. While that’s neat, **it’s not scalable, controllable, or customizable** for enterprise needs.

Our model is:
- **Custom**: Trained from scratch on internal patterns.
- **Personalized**: Mimics specific reviewers' behavior.
- **Extendable**: Can be retrained with company PR data later.
- **Offline-compatible**: Doesn’t need external APIs.

> When we get access to internal PR data in future, this model will become incredibly **powerful, precise, and team-specific**.

---

## 🧠 Tech Stack

| Layer | Tools Used |
|------|------------|
| Model | HuggingFace Transformers `T5ForConditionalGeneration` |
| Tokenization | `T5Tokenizer` |
| Dataset | Custom PyTorch `PRDataset` class |
| Framework | PyTorch, Hugging Face Trainer |
| Training Setup | Fine-tuned on labeled PR comments |
| Deployment | To be pushed to GitHub |
| Optional Inference | Script-based input/output with CSV support |

---

## 🧪 How It Works

1. **Data**: Each training example includes:
   - Reviewer
   - Author
   - File name
   - Category
   - Code review comment

2. **Training**:
   - Custom PyTorch Dataset
   - T5 model fine-tuned over 14 epochs
   - Evaluation at each epoch, best model saved

3. **Inference**:
   - Input PR context → get expected reviewer-style comments
   - Ground truth vs prediction comparison supported

---

## 🔗 Links

- [GitHub Repository] ← (https://github.com/PAND-IAM-IC/AI-PR-REVIEW-SYSTEM)
- [Sample Inference CSV] ← (https://github.com/PAND-IAM-IC/AI-PR-REVIEW-SYSTEM/blob/main/Inference_Results.csv)

---

## 🏁 Future Scope

Once integrated with internal repositories:
- Train on actual PR + review history
- Auto-suggest corrections while writing code
- Build reviewer-specific review bots
- Visualize reviewer patterns

And since it’s our own model—not an API call—we can tweak, inspect, and build further **without any vendor lock-in**.

---

Built for our team, by our team 🚀

