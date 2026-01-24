Natural Language Processing (NLP) is a rapidly growing field that focuses on enabling machines to understand and process human language. Text representation is a crucial aspect of NLP that involves converting raw text data into machine-readable form.

In this article, we will explore the different text representation techniques, starting from traditional approaches such as bag-of-words and n-grams to modern techniques like word embeddings.

By the end of this article, you will have a fair understanding of the different text representation techniques along with their strength and weaknesses.

Press enter or click to view image in full size

Topics to cover:

Bag of words
N-gram
TF-IDF
Word embedding
Sentence embedding
Document embedding
Bag of words (BoW):
This is the simplest way to convert unstructured text data into a structured numeric format that can be processed by machine learning algorithms. Each word in the text is considered a feature, and the number of times a particular word appears in the text is used to represent the importance of that word in the text. Disregarding grammar and word order but keeping track of the frequency of each word.

Example :

Let us consider 3 sentences :

The cat in the hat
The dog in the house
The Bird in the Sky
Press enter or click to view image in full size

Below given code displays the above result:

from sklearn.feature_extraction.text import CountVectorizer
# Sample sentences
sentences = ["The cat in the hat", 
"The dog in the house", "The bird in the sky"]
# Create a CountVectorizer object
vectorizer = CountVectorizer()
# Use the fit_transform method to transform the sentences into a bag of words
bow = vectorizer.fit_transform(sentences)
# Print the vocabulary (features) of the bag of words
print(vectorizer.get_feature_names())
# Print the bag of words
print(bow.toarray())
N-gram:
An N-gram is a traditional text representation technique that involves breaking down the text into contiguous sequences of n-words. A uni-gram gives all the words in a sentence. A Bi-gram gives sets of two consecutive words and similarly, a Tri-gram gives sets of consecutive 3 words, and so on.

Example: The dog in the house

Uni-gram: “The”, “dog”, “in”, “the”, “house”

Become a member
Bi-gram: “The dog”, “dog in”, “in the”, “the house”

Tri-gram: “The dog in”, “dog in the”, “in the house”

TF-IDF:
TF-IDF stands for Term Frequency-Inverse Document Frequency. This is better than BoW since it interprets the importance of a word in a document. The idea behind TF-IDF is to weigh words based on how often they appear in a document (the term frequency) and how common they are across all documents (the inverse document frequency).

The formula for calculating the TF-IDF score of a word in a document is:

Press enter or click to view image in full size

Press enter or click to view image in full size

Below given code displays the above result:

from sklearn.feature_extraction.text import TfidfVectorizer
# Example documents
docs = ["The cat jumped",
        "The white tiger roared",
        "Bird flying in the sky"]  
# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()
# Use the fit_transform method to transform the documents into a TF-IDF matrix
tfidf = vectorizer.fit_transform(docs)
# Print the vocabulary (features) of the TF-IDF matrix
print(vectorizer.get_feature_names())
# Print the TF-IDF matrix
print(tfidf.toarray())
Word embedding:
Word embedding represents each word as a dense vector of real numbers, such that the similar or closely related words are nearer to each other in the vector space. This is achieved by training a neural network model on a large corpus of text, where each word is represented as a unique input and the network learns to predict the surrounding words in the text. The semantic meaning of the word is captured using this. The dimension of these words can range from a few hundred (Glove, Wod2vec)to thousands (Language models).

Below given is the code :

from gensim.models import Word2Vec

# Define the corpus (list of sentences)
corpus = ["The cat jumped",
        "The white tiger roared",
        "Bird flying in the sky"]
corpus=[sent.split(" ") for sent in corpus]
# Train the Word2Vec model on the corpus
model = Word2Vec(corpus, size=50, window=5, min_count=1, workers=2)

# Get the vector representation of a word
vector = model.wv["cat"]
# Get the top-N most similar words to a given word
similar_words = model.wv.most_similar("cat", topn=5)
Press enter or click to view image in full size

Refer document for details : https://jalammar.github.io/illustrated-word2vec/

Sentence Embedding:
It is similar to that of word embedding, the only difference is in place of a word, a sentence is represented as a numerical vector in a high-dimensional space. The goal of sentence embedding is to capture the meaning and semantic relationships between words in a sentence, as well as the context in which the sentence is used.

Below given is the code:

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# Define a list of sentences to be embedded
sentences = ["The cat jumped",
        "The white tiger roared",
        "Bird flying in the sky"]
# Convert the sentences to TaggedDocuments
tagged_data = [TaggedDocument(words=sentence.split(), tags=[str(i)]) for i, sentence in enumerate(sentences)]
# Train a Doc2Vec model on the TaggedDocuments
model = Doc2Vec(tagged_data, vector_size=50, min_count=1, epochs=10)
# Get the embedding for the first sentence
embedding = model.infer_vector("The white tiger roared".split())
# Print the resulting embedding
print(embedding)
Press enter or click to view image in full size

Document Embedding:
Document embedding refers to the process of representing an entire document, such as a paragraph, article, or book, as a single vector. It captures not only the meaning and context of individual sentences but also the relationships and coherence between sentences within the document. The code used is the same as sentence embedding but has multiple sentences within one document. Below given is the code :

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# Define a list of documents to be embedded
documents = ["This is the first document. It has multiple sentences.", 
             "This is the second document. It is shorter than the first.", 
             "This is the third document. It is longer than the first two and has multiple paragraphs."]
# Convert the documents to TaggedDocuments
tagged_data = [TaggedDocument(words=document.split(), tags=[str(i)]) for i, document in enumerate(documents)]
# Train a Doc2Vec model on the TaggedDocuments
model = Doc2Vec(tagged_data, vector_size=50, min_count=1, epochs=10)
# Get the embedding for the first document
embedding = model.docvecs[0]
# Print the resulting embedding
print(embedding)
Press enter or click to view image in full size

In this blog post, we discussed several techniques used for text representation, including Bag of Words, N-grams, TF-IDF, Word Embedding, Sentence Embedding, and Document Embedding. Each technique has its own advantages and disadvantages, and the task at hand determines which is best. Choosing optimal performance in NLP depends on selecting the appropriate technique for a given task.

Thank you for taking the time to read the content.Clap 👏 if you have enjoyed the content.