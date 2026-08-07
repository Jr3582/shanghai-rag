import fitz
import os
import re
import json

data_folder = "../data"
output_folder = "../data/extracted"

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text) #REMOVES [NUMBER] FORMAT IN TEXT
    text = re.sub(r"https?://\S+", "", text) #REMOVES URL'S
    text = re.sub(r"[^\x00-\x7F]+", "", text) #REMOVE NON-ENGLISH CHARCATERS
    text = re.sub(r" {2,}", " ", text) #REMOVES COLLAPS MULTIPLE SPACES
    text = re.sub(r"[""'']", "", text) #REMOVES SMART QUOTES
    text = re.sub(r"\.(:\d+)+", ".", text) #REMOVE :9394:369 TYPE FORMAT

    candidates = [text.find("\nReferences\n"), text.find("\nCitations\n"), text.find("\nNotes\n")]
    valid = [n for n in candidates if n != -1]
    newsletter_index = text.find("\nSign up for our newsletter\n")

    if newsletter_index != -1:
        text = text[:newsletter_index]

    if valid == []:
        return text

    cutoff = min(valid)
    threshold = len(text) * 0.60

    if cutoff > threshold:
        text = text[:cutoff]
   
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    start = 0
    end = chunk_size
    chunks = []
    while start < len(text):
        chunks.append(text[start:end])
        start += chunk_size - overlap
        end += chunk_size - overlap
    return chunks

def chunk_by_sentences(text, max_chunk_size=500):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_chunk = ""
    chunks = []
    for sentence in sentences:
        if len(sentence) >= max_chunk_size:
            current_chunk = current_chunk + " " + sentence
            chunks.append(current_chunk)
            current_chunk = ""
        else:
            if len(current_chunk + sentence) < max_chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk)
                current_chunk = ""
                current_chunk += sentence + " "
    chunks.append(current_chunk)
    return chunks

data = [] #LIST OF DICTIONAIRIES

# Go through every file in the data folder
for filename in os.listdir(data_folder):
    if not filename.endswith("_TRIMMED.pdf"):
        continue  # skip anything that's not a PDF

    filepath = os.path.join(data_folder, filename)
    doc = fitz.open(filepath)

    full_text = ""
    for page in doc:
        full_text += page.get_text()
        full_text += "\n"  # small separator between pages
    
    cleaned_txt = clean_text(full_text)

    chunks = chunk_by_sentences(cleaned_txt)
    for i, chunk in enumerate(chunks):
        dictionary = {
            "source": filename,
            "chunk_id": i,
            "text": chunk
        }
        data.append(dictionary)

    # Save the extracted text as a .txt file with a matching name
    output_name = filename.replace(".pdf", ".txt")
    output_path = os.path.join(output_folder, output_name)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_txt)

    print(f"Extracted {filename} -> {len(cleaned_txt)} characters, {len(doc)} pages")

with open("../data/chunks.json", "w") as f:
    json.dump(data, f, indent=2)
