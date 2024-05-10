import time
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def scibert_model_test(text):

    # Load the SciBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

    # Tokenize the preprocessed and lemmatized text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Measure the time taken to run the model
    start_time = time.time()
    outputs = model(**inputs)
    end_time = time.time()

    print(f"Time taken to run SciBERT on random text: {end_time - start_time:.4f} seconds")
