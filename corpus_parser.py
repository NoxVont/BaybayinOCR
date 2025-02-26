
class CorpusParser:
    def __init__(self, dir, error_threshold=3):
        try:
            with open(dir, 'r', encoding='utf-8') as file:
                self.corpus = file.read().split()
                self.error_threshold = error_threshold
        except Exception:
            raise(f"Exception: Cannot open corpus at {dir}")
    
    def __str__(self):
        return self.corpus
        
    def __len__(self):
        return len(self.corpus)
        
        
    # --- Custom Methods ---
    
    def check_word(self, word: str) -> bool:
        return word.strip().lower() in self.corpus
    
    def correct_word(self, word: str) -> str:
        word = word.lower()
        
        if self.check_word(word):
            return word
        
        import Levenshtein
        
        closest_word = min(self.corpus, key=lambda x: Levenshtein.distance(word, x))
        print(f"CW: {closest_word}")
        if Levenshtein.distance(word, closest_word) <= self.error_threshold:
            print(f"Replaced to {closest_word}")
            return closest_word
        else:
            print("No replacement found.")
            return word
    
    def set_threshold(self, level: int):
        self.error_threshold = level
    
    

# Used for converting raw files into usable format
if __name__ == "__main__":
    corpus = ""

    # Read corpus file
    with open('./corpora/tagalog-news-300K.txt', 'r', encoding='utf-8') as file:
        corpus += file.read()
    
    with open('./corpora/tagalog-wikipedia-100K.txt', 'r', encoding='utf-8') as file:
        corpus += file.read()

    import re
    corpus = re.sub(r'-\s|\s-|\b-\b', '', corpus)   # Clean special characters (except dash)
    corpus = re.sub(r'\d+', '', corpus)             # Clean numbers
    corpus = re.sub(r'\s+', ' ', corpus)            # Clean extra whitespace
    corpus = corpus.lower()                         # Lowercase
    
    with open('./corpora/tagalog-corpus.txt', 'w', encoding='utf-8') as file:
        file.write(corpus)