import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# 1. Load CSV into a Hugging Face Dataset
df = pd.read_csv("cosmic_dataset.csv")  # expects columns: code, E, X, R, W, CFP
# Drop the CFP column since we predict E, X, R, W and recompute CFP
if "CFP" in df.columns:
    df = df.drop(columns=["CFP"])
dataset = Dataset.from_pandas(df)

# 2. Split 80/20 for train/test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
eval_ds  = dataset["test"]

# 3. Initialize CodeBERT tokenizer & model (regression head)
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    problem_type="regression"
)

# 4. Preprocessing: tokenize and attach float32 labels
def preprocess(examples):
    toks = tokenizer(examples["code"], truncation=True, padding=False)
    import numpy as np
    labels = np.vstack([
        examples["E"],
        examples["X"],
        examples["R"],
        examples["W"],
    ]).T.astype(np.float32)
    toks["labels"] = labels.tolist()
    return toks

train_ds = train_ds.map(preprocess, batched=True)
eval_ds  = eval_ds.map(preprocess, batched=True)

# 5. Dynamic padding collator
data_collator = DataCollatorWithPadding(tokenizer)

#6. Compute metrics
def compute_metrics(pred):
    preds, labels = pred
    # preds and labels: shape (n_samples, 4)

    # Per-dimension MSE, RMSE, MAE, R2, Pearson r
    mse_dims  = mean_squared_error(labels, preds, multioutput="raw_values")
    rmse_dims = np.sqrt(mse_dims)
    mae_dims  = mean_absolute_error(labels, preds, multioutput="raw_values")
    r2_dims   = [r2_score(labels[:, i], preds[:, i]) for i in range(4)]
    corr_dims = [pearsonr(labels[:, i], preds[:, i])[0] for i in range(4)]

    # Summed CFP metrics
    sum_preds  = np.sum(preds, axis=1)
    sum_labels = np.sum(labels, axis=1)
    mse_sum    = mean_squared_error(sum_labels, sum_preds)
    rmse_sum   = np.sqrt(mse_sum)
    mae_sum    = mean_absolute_error(sum_labels, sum_preds)
    r2_sum     = r2_score(sum_labels, sum_preds)
    corr_sum   = pearsonr(sum_labels, sum_preds)[0]

    return {
        # Entry
        "mse_E":   mse_dims[0], "rmse_E": rmse_dims[0], "mae_E": mae_dims[0],
        "r2_E":    r2_dims[0],  "corr_E": corr_dims[0],
        # Exit
        "mse_X":   mse_dims[1], "rmse_X": rmse_dims[1], "mae_X": mae_dims[1],
        "r2_X":    r2_dims[1],  "corr_X": corr_dims[1],
        # Read
        "mse_R":   mse_dims[2], "rmse_R": rmse_dims[2], "mae_R": mae_dims[2],
        "r2_R":    r2_dims[2],  "corr_R": corr_dims[2],
        # Write
        "mse_W":   mse_dims[3], "rmse_W": rmse_dims[3], "mae_W": mae_dims[3],
        "r2_W":    r2_dims[3],  "corr_W": corr_dims[3],
        # Total CFP
        "mse_CFP":  mse_sum,    "rmse_CFP": rmse_sum,  "mae_CFP": mae_sum,
        "r2_CFP":   r2_sum,     "corr_CFP": corr_sum
    }

# 7. Training arguments: optimize for total CFP
training_args = TrainingArguments(
    output_dir="codebert-cfp",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="mse_CFP",
    greater_is_better=False
)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
trainer.save_model("codebert-cfp/best-model")

# 10. Display training and validation metrics
print("=== Final training metrics ===")
for key, value in train_result.metrics.items():
    print(f"{key:20s}: {value:.4f}")

print("\n=== Validation set metrics ===")
eval_metrics = trainer.evaluate()
for key, value in eval_metrics.items():
    print(f"{key:20s}: {value:.4f}")
