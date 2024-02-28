import os
import re
from nltk.collocations import TrigramCollocationFinder
import json

# Load texts
texts_original = {}
directory = os.fsencode('./original_langId')
for file in os.listdir(directory):
	filename = os.fsdecode(file)
	directoryname = os.fsdecode(directory)
	with open(directoryname + '/' + filename, 'r', encoding="utf-8") as file:
		data = file.read()
		texts_original[filename] = data

# Preprocess texts
def preprocess_text(text: str) -> str:
	new_text = ""
	sents = text.split('\n')
	for sent in sents:
		sent.strip() # Remove leading and trailing spaces
		sent = sent.lower() # Lowercase
		sent = re.sub(r'[/-*"_+%&@=¬~<>^#»«]', '', sent) # Remove some non-alphabetic characters (not all, to keep some punctuation)
		sent = re.sub(r'\d', '', sent) # Remove digits
		sent = re.sub(r'\s+', ' ', sent) # Remove extra spaces
		new_text += "  " + sent # Add two spaces to separate sentences
	return new_text

texts_new = {}
for file, text in texts_original.items():
	texts_new[file] = preprocess_text(text)
	f = open("./new_langID/" + file, "w", encoding="utf-8")
	f.write(texts_new[file])
	f.close()
	
# Find trigrams
trigrams = {}
unique_chars = {}
for language in texts_new.keys():
	if 'tst' not in language:
		trigram_finder = TrigramCollocationFinder.from_words(texts_new[language])
		tmp = {key : value for key, value in trigram_finder.ngram_fd.items() if value > 5}
		
		trigrams[language] = tmp
		unique_chars[language] = len(set("".join("".join(a) for a in tmp.keys())))

def dict_tuple_to_string(d: dict) -> dict:
	return {key: {"".join(k): v for k, v in value.items()} for key, value in d.items()}

# Save trigrams
with open("./weights/trigrams.json", "w") as trigrams_file:
	json.dump(dict_tuple_to_string(trigrams), trigrams_file)

# Save unique_chars
with open("./weights/unique_chars.json", "w") as unique_chars_file:
	json.dump(unique_chars, unique_chars_file)
	