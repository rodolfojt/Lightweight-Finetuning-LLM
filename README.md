# Lightweight Fine-Tuning with LoRA (DistilBERT)

Parameter-efficient fine-tuning of a binary sentiment classifier using **LoRA (Low-Rank Adaptation)** and Hugging Face **PEFT**.

## Task
Classify Amazon food reviews as:
- `POSITIVE` (Score 4-5)
- `NEGATIVE` (Score 1-3)

## Dataset
- Source: `jhan21/amazon-food-reviews-dataset`
- Subset: shuffled with `seed=42`, first 1,000 samples
- Split: `train_test_split(test_size=0.2)` -> 800 train / 200 test
- Label: `label = 1 if Score >= 4 else 0`
- Text field: `Text`

## Model
- Backbone: `distilbert-base-uncased`
- Sequence classification head: 2 labels (`NEGATIVE`, `POSITIVE`)
- Tokenization: `padding=True`, `truncation=True`

## Baseline (pre-fine-tuning)
The notebook freezes the DistilBERT backbone (`requires_grad=False`) and evaluates with `Trainer`.
- `eval_accuracy`: 0.74
- `eval_loss`: 0.6806

## LoRA fine-tuning (PEFT)
LoRA is configured with:
- `r=16`
- `lora_alpha=8`
- `lora_dropout=0.1`
- `target_modules=["q_lin", "k_lin"]`

Trainable parameters:
- 294,912 trainable out of 67,249,922 (~0.44%)

Trainer settings:
- `learning_rate=2e-3`, `weight_decay=0.01`
- batch sizes: 4 (train/eval)
- epochs: 4
- `evaluation_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`

## Checkpoint handling
A custom `SaveScoreCallback` saves the classifier layer weights at each checkpoint:
- `./model/checkpoint-<step>/score.original_module.pt`

After training, the notebook reloads:
- `./model/checkpoint-800`
and restores classifier weights from `score.original_module.pt`.

## Results
- During training evaluation (`new_trainer.evaluate()`):
  - `eval_accuracy`: 0.82
  - `eval_loss`: 0.4697
- After reloading/restoring (`fine_tuned_trainer.evaluate()`):
  - `eval_accuracy`: 0.75
  - `eval_loss`: 0.6523
- Comparison vs baseline:
  - Accuracy: 0.74 -> 0.75
  - Loss: 0.6806 -> 0.6523
  - Eval runtime: 3.9743s -> 3.4142s

## How to run
Open `LightweightFineTuning.ipynb` and run cells top-to-bottom.
The notebook will write checkpoints and artifacts to `./model/`.

## References
- Udacity Generative AI Nanodegree: BERT sentiment classifier exercises
- Udacity Generative AI Nanodegree: Full fine-tuning BERT
