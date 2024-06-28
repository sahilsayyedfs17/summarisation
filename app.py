from flask import Flask, request, render_template, redirect, url_for, flash
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
from nltk.tokenize import sent_tokenize
import nltk

# Ensure nltk resources are available
nltk.data.path.append('./nltk_data')

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the local model
model_path = "./local_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            summary = summarize_text(filepath)
            return render_template('index.html', summary=summary)
        except Exception as e:
            flash(f'An error occurred: {e}')
            return redirect(request.url)
    return redirect(request.url)

def simplify_text(text):
    sentences = sent_tokenize(text)
    simplified_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\b(complex|complicated|intricate|sophisticated)\b', 'simple', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\b(approximately|around|about)\b', 'about', sentence, flags=re.IGNORECASE)
        simplified_sentences.append(sentence)
    return ' '.join(simplified_sentences)

def summarize_text(filepath):
    with open(filepath, 'r') as file:
        text = file.read()

    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            simplified_summary = simplify_text(summary)
            summaries.append(simplified_summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    
    full_summary = ' '.join(summaries)
    
    bullet_points = convert_to_bullet_points(full_summary)
    
    return bullet_points

def convert_to_bullet_points(text):
    sentences = sent_tokenize(text)
    bullet_points = ['• ' + sentence for sentence in sentences]
    return '\n'.join(bullet_points)

if __name__ == '__main__':
    app.run(debug=True)
