import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

chat_words = {
    "A3": "Anytime, Anywhere, Anyplace", "ADIH": "Another Day In Hell",
    "AFK": "Away From Keyboard", "AFAIK": "As Far As I Know",
    "ASAP": "As Soon As Possible", "ASL": "Age, Sex, Location",
    "ATM": "At The Moment", "BAE": "Before Anyone Else",
    "BBL": "Be Back Later", "BRB": "Be Right Back", "BRUH": "Bro",
    "BTW": "By The Way", "DM": "Direct Message", "FAQ": "Frequently Asked Questions",
    "FOMO": "Fear Of Missing Out", "FR": "For Real", "FYI": "For Your Information",
    "GG": "Good Game", "GOAT": "Greatest Of All Time", "IDK": "I Don't Know",
    "ILY": "I Love You", "IMO": "In My Opinion", "IMHO": "In My Honest Opinion",
    "IRL": "In Real Life", "JK": "Just Kidding", "LMAO": "Laughing My Ass Off",
    "LOL": "Laughing Out Loud", "NVM": "Never Mind", "OMG": "Oh My God",
    "POV": "Point Of View", "ROFL": "Rolling On The Floor Laughing",
    "RN": "Right Now", "SUS": "Suspicious", "TBH": "To Be Honest",
    "THX": "Thank You", "TLDR": "Too Long Didn't Read", "TTYL": "Talk To You Later",
    "U": "You", "U2": "You Too", "R": "are", "W": "Win", "WTF": "What The Fuck",
    "WTG": "Way To Go", "ZZZ": "Sleeping Bored Tired"
}


def chat_words_process(text):
    new_text = []
    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text).lower()


def Text_procecess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r'[^\w\s]', '', text)

    # Chat words
    text = chat_words_process(text)

    # Tokenization
    token = word_tokenize(text)

    # Lemmatization
    text = [lemma.lemmatize(w, pos='v') for w in token]

    # Stopword removal
    text = [w for w in text if w not in stop_words]

    return " ".join(text)