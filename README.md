# T5 Summarizer — Fine-tune

A small project/notebook for fine-tuning T5 models (t5-small / t5-base / t5-large) to perform abstractive summarization. This repository demonstrates data preparation, training, inference and evaluation workflows using Hugging Face Transformers, Datasets and common summarization metrics (ROUGE).

## Table of Contents
- About
- Repository structure (recommended)
- Requirements
- Installation
- Data format & preparation
- Fine-tuning (examples)
- Inference (use the fine-tuned model)
- Evaluation
- Tips & best practices
- Contributing
- License

## About
T5 (Text-To-Text Transfer Transformer) is a flexible text-to-text model that can be fine-tuned to perform summarization by training it to map articles to summaries. This repository contains examples and guidance to fine-tune T5 on your summarization dataset and to run generation and evaluation.

## Recommended repository structure
(Adjust paths to match your repository layout)
- notebooks/
  - train.ipynb           — a notebook that performs full training example
  - inference.ipynb       — example inference and sample generation
  - eval.ipynb            — evaluation using ROUGE
- data/
  - train.csv             — recommended: columns: `article`, `summary`
  - val.csv
  - test.csv
- scripts/
  - train.py              — optional script to fine-tune with Hugging Face
  - infer.py
- outputs/
  - t5-finetuned/         — model checkpoints and tokenizer
- README.md
- requirements.txt

## Requirements
- Python 3.8+
- PyTorch (or Flax if you prefer)
- transformers >= 4.x
- datasets
- sentencepiece
- tokenizers
- evaluate (or rouge_score)
- accelerate (optional, recommended for multi-GPU / mixed precision)

Example minimal pip install:
pip install torch transformers datasets sentencepiece evaluate accelerate

Or, using requirements.txt:
pip install -r requirements.txt

## Data format & preparation
This repository expects a summarization dataset with at least two columns:
- `article` (the long text to summarize)
- `summary` (the target short summary)

Acceptable container formats: CSV, JSON, or Hugging Face Dataset format. Example CSV row:
"article","summary"
"Long article text...","A short summary..."

Loading with datasets (example):
from datasets import load_dataset
ds = load_dataset('csv', data_files={'train': 'data/train.csv', 'validation': 'data/val.csv'})

Tokenization / preprocessing example (T5):
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def preprocess(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)

## Fine-tuning examples

### Option A — Use Hugging Face examples (recommended quick start)
The Transformers repository includes example scripts like `run_seq2seq.py`. Example command:
python run_seq2seq.py \
  --model_name_or_path t5-base \
  --dataset_name csv \
  --train_file data/train.csv \
  --validation_file data/val.csv \
  --source_prefix "summarize: " \
  --output_dir outputs/t5-finetuned \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --predict_with_generate \
  --max_source_length 512 \
  --max_target_length 128 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --logging_steps 100 \
  --save_strategy epoch

(Adapt flags to your script or notebook. If using a notebook, the same parameters are used inside the training cell.)

### Option B — Use Seq2SeqTrainer in a script
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

training_args = Seq2SeqTrainingArguments(
    output_dir="outputs/t5-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True  # if supported
)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer
)
trainer.train()

## Inference (generate summaries)
Load model and tokenizer and run generate:
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("outputs/t5-finetuned")
model = AutoModelForSeq2SeqLM.from_pretrained("outputs/t5-finetuned")

input_text = "summarize: " + long_article_text
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
output_ids = model.generate(
    inputs.input_ids,
    num_beams=4,
    max_length=128,
    early_stopping=True,
    repetition_penalty=2.0
)
generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated)

## Evaluation
Typical metric: ROUGE (ROUGE-1, ROUGE-2, ROUGE-L). Using `evaluate` (huggingface):
import evaluate
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=preds, references=refs)
print(results)

Tips:
- Use `predict_with_generate=True` for the trainer to obtain generated predictions during evaluation.
- Post-process generated summaries (strip whitespace, remove broken tokens) before computing ROUGE.

## Tips & best practices
- Prefix inputs with "summarize: " (T5-style prompts) for better results.
- Keep target lengths reasonable (e.g., max_target_length=128).
- Use gradient accumulation if you have limited GPU memory.
- Mixed precision (fp16) speeds up training and reduces memory usage.
- Use learning rate scheduling and early stopping if possible.
- Experiment with t5-small → t5-base → t5-large depending on resources.
- Consider using the Hugging Face Hub to share and version models: `model.push_to_hub(...)`.

## Reproducibility & checkpoints
- Save checkpoints frequently and log hyperparameters.
- Seed your experiments for reproducibility: set random seeds for `numpy`, `torch`, and `random`.

## Contributing
Contributions, issues and feature requests are welcome. Please:
1. Open an issue describing the change or bug.
2. Submit a pull request with a clear description and tests or a notebook demonstrating the change.

## Add a license
This repository currently doesn't include a license file. If you want others to reuse your code, add a LICENSE (MIT, Apache-2.0, etc.).

## Contact
Created by @nna0921 — feel free to open issues or reach out via GitHub.

Happy fine-tuning!
