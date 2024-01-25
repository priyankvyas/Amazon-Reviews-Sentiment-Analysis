from nltk.stem import PorterStemmer
import re
import string
from nltk.corpus import stopwords

# Initialize the PorterStemmer for stemming words
ps = PorterStemmer()

# Preprocess the DataFrame by removing links, punctuations, and blank spaces, converting all the letters
# to lowercase, and stemming any words that are not included in the stopwords list from the review text
def preprocess(df):
    for index, row in df.iterrows():
        text = str(row["reviewText"]) + " " + str(row["summary"])
        linkRemovedText = re.sub(r"(<a.+>)|(value=http.+)|(http.+)", "", text.lower())
        lowerAndUnpuncText = linkRemovedText.translate(str.maketrans('', '', string.punctuation))
        blankSpaceRemovedText = re.findall(r'\w+', lowerAndUnpuncText)
        stemmedText = ""
        for word in blankSpaceRemovedText:
            if word not in stopwords.words('english'):
                stemmedText += ps.stem(word) + " "
        df.loc[index, "preprocessedText"] = stemmedText[:-1]
    return df