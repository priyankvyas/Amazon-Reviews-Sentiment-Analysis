from nltk.stem import PorterStemmer
import re
import string

ps = PorterStemmer()

def preprocess(text):
    linkRemovedText = re.sub(r"(<a.+>)|(value=http.+)|(http.+)", "", text.lower())
    blankSpaceRemovedText = re.sub(r"\s+", " ", linkRemovedText.replace('\n', ' '))
    lowerAndUnpuncText = blankSpaceRemovedText.translate(str.maketrans('', '', string.punctuation))
    stemmedText = ""
    for word in lowerAndUnpuncText.split(' '):
        stemmedText += ps.stem(word) + " "
    return stemmedText[:-1]