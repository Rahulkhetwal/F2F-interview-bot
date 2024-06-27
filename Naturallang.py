## TOKENIZATION  [ breaking text into smaller units typically words or sentences]

from nltk.tokenize import word_tokenize, sent_tokenize

sentence = "NLTK is a leading platform for building Python programs to work with human language data."

# Tokenize words
words = word_tokenize(sentence)
print(words)

# Tokenize sentences
sentences = sent_tokenize(sentence)
print(sentences)


## PART OF SPEECH (POS)TAGGING   [provide understandable tags to the words for better understanding]

from nltk import pos_tag

# Perform POS tagging
tagged_words = pos_tag(words)
print(tagged_words)




##  NAMED ENTITY RECOGNITION (NER): NER involves identifying and classifying named entities (e.g., persons, organizations, locations) in text. NLTK provides functions for NER:

from nltk import ne_chunk

# Perform NER
ner_tags = ne_chunk(tagged_words)
print(ner_tags)
##  CHUNKING  [grouping words into meaningful chunks]



## CHUNKING  AND  PARSING  []

from nltk.chunk import RegexpParser

# Define a chunk grammar
grammar = r"""
    NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}                # Chunk prepositions followed by NP
    VP: {<VB.*><NP|PP|CLAUSE>+$}  # Chunk verbs and their arguments
    CLAUSE: {<NP><VP>}            # Chunk NP, VP pairs
"""

# Create a chunk parser
chunk_parser = RegexpParser(grammar)

# Apply chunking
chunks = chunk_parser.parse(tagged_words)
print(chunks)

