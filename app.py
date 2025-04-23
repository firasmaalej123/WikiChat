import requests
import spacy
from bs4 import BeautifulSoup
from transformers import pipeline
import torch
from textblob import TextBlob
torch.set_num_threads(4)  # Adjust based on your CPU cores

grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
def correct_spelling(text):
    """Corrects spelling mistakes using TextBlob."""
    return str(TextBlob(text).correct())

def correct_grammar(text):
    """Corrects grammatical errors using a transformer-based model."""
    try:
        corrected = grammar_corrector(text, max_new_tokens=100)[0]['generated_text']
        return corrected
    except Exception as e:
        print("Grammar correction failed:", e)
        return text
def robust_text_correction(text):
    # First, do a quick spelling fix with TextBlob (or similar)
    text = correct_spelling(text)
    # Then, refine grammar with the transformer model
    text = correct_grammar(text)
    return text

# Load NLP model
nlp = spacy.load("en_core_web_sm")


def generate_search_query(user_input):
    """Extracts key entities and relevant words from user input while tolerating grammatical errors."""
    corrected_input = robust_text_correction(user_input)  # Spell correction
    doc = nlp(corrected_input)

    # Extract meaningful words (proper nouns, nouns, verbs, adjectives, numbers)
    keywords = [
        token.lemma_ for token in doc 
        if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "NUM"} and not token.is_stop
    ]

    # Extract named entities
    entities = [ent.text for ent in doc.ents]

    # Combine keywords and entities while preserving order
    unique_terms = list(dict.fromkeys(keywords + entities))

    search_query = " ".join(unique_terms)
    
    return search_query if len(search_query) >= 3 else corrected_input.strip()
def search_wikipedia(query):
    """Fetches the first Wikipedia page URL based on the search query."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None  # Handle errors
    
    data = response.json()
    results = data.get("query", {}).get("search", [])
    
    if not results:
        return None
    
    page_title = results[0]["title"].replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{page_title}"

def extract_text_from_page(url):
    """Fetches and extracts full text content from a Wikipedia page."""
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return "Failed to retrieve content."
        
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  # Extract paragraph text
        
        text_content = " ".join([p.get_text() for p in paragraphs if p.get_text().strip()])
        
        # Remove Wikipedia citations like [1], [2], etc.
        cleaned_text = " ".join(text_content.split())
        return cleaned_text
    
    except requests.RequestException:
        return "Failed to retrieve content."

# Load a pre-trained summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, max_tokens=1000):
    """Splits text into chunks that the model can process."""
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def summarize_content(content):
    if len(content) < 50:
        return content
    chunks = chunk_text(content, max_tokens=400)
    summaries = summarizer(chunks, max_length=100, min_length=50, do_sample=False)
    return " ".join([s['summary_text'] for s in summaries])
def main():
    print("👋 Welcome! Ask me anything, and I'll fetch a quick summary for you! (Type 'exit' to quit)\n")

    while True:
        user_input = input("💡 Your query: ").strip()

        if not user_input:
            print("⚠️ Please enter a valid query!")
            continue
        
        if user_input.lower() == 'exit':
            print("👋 Exiting... Have a great day!")
            break
        
        

        search_query = generate_search_query(user_input)
        print("\n🔍 Searching for:", search_query)
        wikipedia_url = search_wikipedia(search_query)

        if wikipedia_url:
            print(f"📖 Found a source: {wikipedia_url}")
            extracted_text = extract_text_from_page(wikipedia_url)
            summarized_text = summarize_content(extracted_text)

            print("\n📌 **Quick Summary:**\n", summarized_text)
            print("\n🔗 **Source:**", wikipedia_url)
        else:
            print("❌ No Wikipedia results found for this query. Try rephrasing!")

        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()
