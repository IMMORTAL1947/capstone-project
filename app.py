import pandas as pd
from transformers import GPT2Tokenizer

# Load the dataset
file_path = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
data = pd.read_csv(file_path)

# Remove irrelevant columns
data = data[['instruction', 'response']]

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the instructions and responses
data['instruction_tokenized'] = data['instruction'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=50))
data['response_tokenized'] = data['response'].apply(lambda x: tokenizer.encode(x, truncation=True, padding='max_length', max_length=50))

# Display the tokenized data
print(data.head())

from sklearn.model_selection import train_test_split

# Split the data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Display the sizes of the training and validation sets
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

import torch

# Define a custom dataset class
class CustomerSupportDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create the training and validation datasets
train_dataset = CustomerSupportDataset(
    encodings={'input_ids': train_data['instruction_tokenized'].tolist()},
    labels=train_data['response_tokenized'].tolist()
)
val_dataset = CustomerSupportDataset(
    encodings={'input_ids': val_data['instruction_tokenized'].tolist()},
    labels=val_data['response_tokenized'].tolist()
)


from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Load the pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()


# Evaluate the model
eval_results = trainer.evaluate()

# Display evaluation results
print(f"Evaluation results: {eval_results}")

# Function to generate responses
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

sample_instruction = "I need help with my order"
print(f"Generated response: {generate_response(sample_instruction)}")

