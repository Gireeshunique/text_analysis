import requests
import pandas as pd
import re
import math
import pyphen
from bs4 import BeautifulSoup
from pathlib import Path

# ------------------------------------------------------------------
# LOAD DATA FILES
# ------------------------------------------------------------------
base = Path(".")
input_file = base / "Input.xlsx"
output_file = base / "Output Data Structure.xlsx"
stopwords_dir = base / "StopWords"
dict_dir = base / "MasterDictionary"

df = pd.read_excel(input_file)

# ------------------------------------------------------------------
# LOAD STOPWORDS
# ------------------------------------------------------------------
stopwords = set()
for file in stopwords_dir.rglob("*.txt"):
    for line in open(file, "r", encoding="latin-1"):
        sw = line.strip().lower()
        if sw:
            stopwords.add(sw)

# ------------------------------------------------------------------
# LOAD POSITIVE & NEGATIVE WORDS
# ------------------------------------------------------------------
def load_dict(path):
    words = set()
    for line in open(path, "r", encoding="latin-1"):
        w = line.strip().lower()
        if w and not w.startswith(";"):
            words.add(w)
    return words

positive_words = load_dict(dict_dir / "positive-words.txt")
negative_words = load_dict(dict_dir / "negative-words.txt")

# ------------------------------------------------------------------
# TEXT TOKENIZATION HELPERS
# ------------------------------------------------------------------
dic = pyphen.Pyphen(lang='en')

def syllable_count(word):
    return dic.inserted(word).count("-") + 1

def tokenize_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z']", " ", text)
    return [w for w in text.split() if w]

def split_sentences(text):
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]

# ------------------------------------------------------------------
# TEXT ANALYSIS FUNCTION
# ------------------------------------------------------------------
def analyze_text(text):
    sentences = split_sentences(text)
    sent_count = max(len(sentences), 1)

    tokens = tokenize_words(text)
    content_tokens = [w for w in tokens if w not in stopwords]
    word_count = len(content_tokens)

    pos_score = sum(1 for w in content_tokens if w in positive_words)
    neg_score = sum(1 for w in content_tokens if w in negative_words)

    polarity = (pos_score - neg_score) / ((pos_score + neg_score) or 1e-6)
    subjectivity = (pos_score + neg_score) / (word_count or 1e-6)

    complex_words = [w for w in content_tokens if syllable_count(w) >= 3]
    pct_complex = len(complex_words) / (word_count or 1e-6)

    total_syllables = sum(syllable_count(w) for w in content_tokens)
    syll_per_word = total_syllables / (word_count or 1e-6)

    avg_sentence_len = len(tokens) / sent_count
    fog_index = 0.4 * (avg_sentence_len + pct_complex * 100)

    personal_pronouns = len(re.findall(r"\b(I|we|my|ours|us)\b", text, flags=re.I))
    avg_word_len = sum(len(w) for w in content_tokens) / (word_count or 1e-6)

    return [
        pos_score, neg_score, polarity, subjectivity, avg_sentence_len,
        pct_complex, fog_index, avg_sentence_len, len(complex_words),
        word_count, syll_per_word, personal_pronouns, avg_word_len
    ]

# ------------------------------------------------------------------
# SCRAPE, ANALYZE & SAVE
# ------------------------------------------------------------------
results = []

for index, row in df.iterrows():
    url_id = row["URL_ID"]
    url = row["URL"]

    print(f"Processing URL_ID {url_id} ...")

    try:
        page = requests.get(url, timeout=15)
        soup = BeautifulSoup(page.text, "html.parser")

        # extract article text
        article = " ".join([p.text for p in soup.find_all("p")]).strip()

        if len(article) < 50:
            raise Exception("Article too short or missing")

        metrics = analyze_text(article)

    except Exception as e:
        print(f"❌ Error in {url_id}, inserting zeros")
        metrics = [0] * 13

    results.append([url_id] + metrics)

# save
cols = [
    "URL_ID", "POSITIVE SCORE", "NEGATIVE SCORE", "POLARITY SCORE",
    "SUBJECTIVITY SCORE", "AVG SENTENCE LENGTH",
    "PERCENTAGE OF COMPLEX WORDS", "FOG INDEX",
    "AVG NUMBER OF WORDS PER SENTENCE", "COMPLEX WORD COUNT",
    "WORD COUNT", "SYLLABLE PER WORD",
    "PERSONAL PRONOUNS", "AVG WORD LENGTH"
]

out_df = pd.DataFrame(results, columns=cols)
out_df.to_excel(output_file, index=False)

print("\n🎉 COMPLETED — Results written to Output Data Structure.xlsx")
