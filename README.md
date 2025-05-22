
#  BERT-Based Multi-Label Text Classification

This project implements a **multi-label text classification** system using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model with PyTorch and HuggingFace Transformers.

---

## 📌 Project Overview

- Uses **`bert-base-uncased`** from HuggingFace.
- Designed for **multi-label classification** (6 output classes).
- Tokenizes and encodes text data using `BertTokenizer`.
- Handles training and validation with checkpointing.
- Inference with sigmoid activation for multi-label outputs.

---

## 📁 Project Structure

```
.
├── CustomDataset class     # Prepares input data for BERT
├── BERTClass               # Defines model architecture
├── train_model()          # Handles training, validation, checkpointing
├── Inference block         # Runs predictions on new text
├── save_ckp()              # Saves model checkpoints
└── 5689a604-b666...ipynb   # Full Jupyter Notebook
```

---

## 🔧 Setup & Requirements

Install required packages:

```bash
pip install torch transformers pandas numpy scikit-learn
```

---

## 📚 Dataset Format

The input DataFrame should contain:

- A column `context` (text data).
- Target columns defined in `target_list` (6 labels for classification).

---

## 🧱 Model Architecture

```python
class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(768, 6)  # 6 output labels

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.bertmodel(input_ids, attention_mask, token_type_ids)
        output_dropout = self.dropout(output_1.pooler_output)
        output = self.linear(output_dropout)
        return output
```

---

## 🧪 Custom Dataset Class

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len
        self.title = self.df['context']
        self.targets = self.df[target_list].values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets': torch.FloatTensor(self.targets[index])
        }
```

---

## 🏋️ Training the Model

```python
model = BERTClass().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-5)
loss_fn = nn.BCEWithLogitsLoss()

model = train_model(
    n_epochs=3,
    training_loader=train_loader,
    validation_loader=val_loader,
    model=model,
    optimizer=optimizer,
    checkpoint_path='checkpoint.pt',
    best_model_path='best_model.pt'
)
```

---

## 🧠 `train_model()` Function

```python
def train_model(n_epochs, training_loader, validation_loader, model, optimizer, checkpoint_path, best_model_path):
    valid_loss_min = np.inf

    for epochs in range(1, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        for index, batch in enumerate(training_loader):
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.float)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += (1 / (index + 1)) * (loss.item() - train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(validation_loader):
                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                targets = batch['targets'].to(device, dtype=torch.float)

                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs, targets)

                valid_loss += (1 / (index + 1)) * (loss.item() - valid_loss)

        checkpoint = {
            'epoch': epochs + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    return model
```

---

## 🧪 Inference

```python
encodings = tokenizer.encode_plus(
    example_text,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    return_token_type_ids=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

model.eval()
with torch.no_grad():
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    token_type_ids = encodings['token_type_ids'].to(device)

    outputs = model(input_ids, attention_mask, token_type_ids)
    final_output = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
    print(final_output)
```

---

## 💾 Checkpointing

Model and optimizer states are saved at each epoch:

```python
checkpoint = {
    'epoch': current_epoch,
    'valid_loss_min': current_val_loss,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
}
```

Use the `save_ckp()` function to save to `checkpoint.pt` and `best_model.pt`.

---

## ✍️ Author

Built by Twaha 
Cridit: https://www.youtube.com/watch?v=f-86-HcYYi8&t=2263s 
Feel free to contribute or raise an issue!
