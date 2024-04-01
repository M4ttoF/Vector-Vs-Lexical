import nltk
nltk.download('brown')

# Get the list of genres (categories) in the Brown Corpus
genres = nltk.corpus.brown.categories()

# Print the list of genres
print("Genres in the Brown Corpus:")
print(genres)
