import pandas as pd

df = pd.read_csv("bios.csv")  # Your file
df = df.dropna()  # Drop empty bios

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(example):
    return tokenizer(example['bio_text'], padding='max_length', truncation=True, max_length=64)

from datasets import Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
val_dataset = dataset['test']


from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_dir='./logs',
    load_best_model_at_end=True
)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

trainer.evaluate()


text = "Earn money in 3 hours by clicking the link!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
output = model(**inputs)
pred = output.logits.argmax().item()

print("Prediction:", "Fake" if pred == 1 else "Real")
