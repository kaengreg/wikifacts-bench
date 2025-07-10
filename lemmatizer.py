import spacy
import spacy.cli
import subprocess
import sys
from spacy.language import Language
import pymorphy3

class Pymorphy3Lemmatizer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def __call__(self, doc):
        for token in doc:
            if token.is_alpha:
                token.lemma_ = self.morph.parse(token.text)[0].normal_form
        return doc

@Language.factory("pymorphy_lemmatizer")
def create_pymorphy_lemmatizer(nlp, name):
    return Pymorphy3Lemmatizer()

class MultilingualLemmatizer:
    def __init__(self, lang: str):
        model_news = f"{lang}_core_news_sm"
        model_web  = f"{lang}_core_web_sm"

        try:
            self.nlp = spacy.load(model_news)
        except OSError:
            try:
                self.nlp = spacy.load(model_web)
            except OSError:
                print(f"Neither {model_news} nor {model_web} found. Downloading {model_news}…")
                try:
                    spacy.cli.download(model_news)
                    self.nlp = spacy.load(model_news)
                except Exception:
                    print(f"Failed to download {model_news}. Trying download of {model_web}…")
                    try:
                        spacy.cli.download(model_web)
                        self.nlp = spacy.load(model_web)
                    except Exception:
                        print(f"Failed to download both models. Using blank pipeline.")
                        self.nlp = spacy.blank(lang)
                        self.nlp.add_pipe('attribute_ruler')
                        self.nlp.add_pipe('lemmatizer', config={'mode': 'rule'})
                        self.nlp.initialize()

        if lang.lower() in ('ru', 'uk'):
            self.nlp.add_pipe('pymorphy_lemmatizer')


    def lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text)
        lemmas = [tok.lemma_ for tok in doc if (tok.is_alpha or tok.is_digit) and len(tok) > 1]
        return ' '.join(lemmas)