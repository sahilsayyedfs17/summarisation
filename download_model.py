from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Specify the model name
model_name = "facebook/bart-large-cnn"

# Download and save the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Specify the directory to save the model
save_directory = "./local_model"

# Save the model and tokenizer
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
