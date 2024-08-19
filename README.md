# Fine-Tuning GPT-2 for Custom Text Generation

This project walks you through fine-tuning the GPT-2 model, a powerful transformer model developed by OpenAI, on a custom text dataset. You'll learn how to prepare the dataset, train the model, and generate text based on user prompts using your fine-tuned model.

## 1. Setup and Installation

### Install Required Libraries

To start, you'll need to install the necessary libraries:

```bash
!pip install transformers
```

The `transformers` library from Hugging Face provides the tools to work with pre-trained models like GPT-2. 

### Import Modules

After installation, import the essential modules for the project:

```python
import math
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
```

These imports include the `Trainer` and `TrainingArguments` classes for training, and other components necessary for working with the GPT-2 model and dataset.

## 2. Dataset Preparation and Model Training

### Load and Prepare the Dataset

Define functions to load your dataset and prepare it for training:

```python
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator
```

The `load_dataset` function loads and tokenizes your dataset, while `load_data_collator` prepares it for language modeling tasks.

### Fine-Tune the GPT-2 Model

The core training function handles the fine-tuning of GPT-2:

```python
def train(train_file_path, eval_file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    eval_dataset = load_dataset(eval_file_path, tokenizer) if eval_file_path else None
    data_collator = load_data_collator(tokenizer)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
        logging_dir='./logs',
        logging_steps=500,
        evaluation_strategy="epoch" if eval_dataset else "no",
        learning_rate=3e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    if eval_dataset:
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results['eval_loss'])
        print(f'Perplexity: {perplexity}')

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
```

### Training Parameters

Define the paths and parameters for your training process:

```python
train_file_path = "/content/Music_Data_CLEANED[standardized].txt"
eval_file_path = "/content/eval.txt"
model_name = 'gpt2-medium'
output_dir = '/content'
overwrite_output_dir = True
per_device_train_batch_size = 16
num_train_epochs = 10
save_steps = 15
```

Finally, initiate the training process:

```python
train(
    train_file_path=train_file_path,
    eval_file_path=eval_file_path,
    model_name=model_name,
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps
)
```

This will fine-tune the GPT-2 model on your dataset and save the trained model and tokenizer in the specified output directory.

## 3. Text Generation with the Trained Model

### Load the Model and Tokenizer

After training, load the fine-tuned model and tokenizer:

```python
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer
```

### Generate Text

Define the function for generating text using your model:

```python
def generate_text(model, tokenizer, sequence, max_length):
    ids = tokenizer.encode(sequence, return_tensors='pt')

    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        top_k=50,
        top_p=0.95,
    )

    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return generated_text
```

### Interactive Prompt

Allow users to interactively generate text:

```python
model_path = "/content"
tokenizer_path = model_path

model = load_model(model_path)
tokenizer = load_tokenizer(tokenizer_path)

print("Type 'exit' to quit the loop.")
while True:
    user_input = input("\nEnter your prompt: ")
    if user_input.strip().lower() == 'exit':
        print("Exiting the loop. Goodbye!")
        break

    max_len = 100
    response = generate_text(model, tokenizer, user_input, max_len)

    print("\n--- Generated Response ---")
    print(response)
    print("\n---------------------------")
```

This loop allows users to input prompts and receive generated responses from the fine-tuned model.

## 4. Conclusion and Further Steps

### Key Considerations

- **Training Settings**: You may need to adjust parameters like `num_train_epochs` and `per_device_train_batch_size` depending on the size of your dataset and the computational resources available.
- **Text Generation**: Experiment with parameters like `max_length`, `top_k`, and `top_p` to fine-tune the quality of generated text.

### Troubleshooting

- **Memory Issues**: Lower batch sizes or reduce the sequence length if you encounter memory errors.
- **Model Performance**: Ensure your dataset is properly formatted and preprocessed. Adjust learning rates and training epochs as needed.

### Next Steps

Explore using the fine-tuned model for specific applications such as dialogue generation, creative writing, or other language tasks. Consider sharing your fine-tuned model and results with me at https://discord.com/invite/6XARWzh9, or email me at stephen07.nt.work@gmail.com for collaboraions, or to just connect. 
PEACEOUT!
