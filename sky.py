# 1) Morphology is the study of the way words are built up from smaller meaning bearing units. 

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

nltk.download('punkt')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

text = "The cats are happily running towards the biggest playground."

tokens = word_tokenize(text)
print("Tokens:", tokens)

stemmer = PorterStemmer()
stems = [stemmer.stem(token) for token in tokens]
print("Stemming:", stems)

lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(token.lower(), pos='v') for token in tokens]
print("Lemmatization:", lemmas)

doc = nlp(text)
print("\nMorphological Analysis:")
for token in doc:
    print(f"{token.text}: {token.lemma_} ({token.pos_})")





# 2)Study and understand the concepts of Morphology by the use of add, delete table. 
   
morph_table = [
    {"Base Word": "play", "Prefix": "re", "Suffix": "ing"},
    {"Base Word": "happy", "Prefix": "un", "Suffix": ""},
    {"Base Word": "teach", "Prefix": "", "Suffix": "er"}
]

for row in morph_table:
    base = row["Base Word"]
    prefix = row["Prefix"]
    suffix = row["Suffix"]
    new_word = prefix + base + suffix
    row["New Word"] = new_word

print("Table After Add Operation:\n")
for row in morph_table:
    print(row)

print("\nTable After Delete Operation:\n")
for row in morph_table:
    new_word = row["New Word"]
    prefix = row["Prefix"]
    suffix = row["Suffix"]

    word_after_prefix_removal = new_word[len(prefix):] if prefix and new_word.startswith(prefix) else new_word
    word_after_suffix_removal = word_after_prefix_removal[:-len(suffix)] if suffix and new_word.endswith(suffix) else word_after_prefix_removal
    
    print(f"Original: {new_word} ➡ Base: {word_after_suffix_removal}")






# 3)Implement Part-Of-Speech tagging

import spacy
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog."
doc = nlp(text)
for token in doc:
    print(f"{token.text}: {token.pos_}")





# 4)Identify Semantic Relation between words (using wordnet).

import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
nltk.download('omw-1.4')

def get_relationships(w1, w2):
    s1 = wn.synsets(w1)
    s2 = wn.synsets(w2)
    if not s1 or not s2:
        return f"Can't find meanings for '{w1}' or '{w2}'."
    
    s1 = s1[0]
    s2 = s2[0]
    
    return {
        "Word 1": w1,
        "Word 2": w2,
        "Definition 1": s1.definition(),
        "Definition 2": s2.definition(),
        "Similarity": s1.wup_similarity(s2),
        "Hypernyms 1": [l.name() for h in s1.hypernyms() for l in h.lemmas()],
        "Hypernyms 2": [l.name() for h in s2.hypernyms() for l in h.lemmas()],
        "Hyponyms 1": [l.name() for h in s1.hyponyms() for l in h.lemmas()],
        "Hyponyms 2": [l.name() for h in s2.hyponyms() for l in h.lemmas()],
    }

rel = get_relationships("dog", "cat")
for k, v in rel.items():
    print(f"{k}: {v}")






# 5)Implement N-Gram Model 

from collections import defaultdict

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(lambda: defaultdict(int))
    
    def train(self, texts):
        for text in texts:
            words = ['<START>'] + text.lower().split() + ['<END>']
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i + self.n - 1])
                target = words[i + self.n - 1]
                self.ngrams[context][target] += 1
    
    def predict(self, context):
        context = tuple(context)
        if context in self.ngrams:
            return max(self.ngrams[context].items(), key=lambda x: x[1])[0]
        return None

# Example usage
model = NGramModel(2)  # Bigram model
texts = ["I love natural language processing", "natural language models are fascinating"]
model.train(texts)
print(f"After 'natural', predicted: {model.predict(['natural'])}")









# 6) Build & Evaluate a NER system using existing NER libraries  

import spacy
from sklearn.metrics import classification_report

def predict_entities(text):
    return [(ent.text, ent.label_) for ent in spacy.load("en_core_web_sm")(text).ents]

def evaluate_ner(test_data):
    true_labels, pred_labels = [], []
    for text, true_ents in test_data:
        preds = predict_entities(text)
        true_labels.extend([label for _, label in true_ents])
        pred_labels.extend([label for _, label in preds])
    return classification_report(true_labels, pred_labels)

text = "Apple CEO Tim Cook announced new iPhone models in California yesterday."
print("\nEntities:", predict_entities(text))


test_data = [("Microsoft's Satya Nadella visited London.", [("Microsoft", "ORG"), ("Satya Nadella", "PERSON"), ("London", "GPE")]),
             ("Google opened a new office in Paris.", [("Google", "ORG"), ("Paris", "GPE")])]

print("\nEvaluation:", evaluate_ner(test_data))







# 7) Perform  Named Entity Recognition on a given text

import spacy

nlp = spacy.load("en_core_web_sm")

def ner(text):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]

text = "Barack Obama was born in Hawaii and served as the 44th President of the United States."

print(ner(text))
