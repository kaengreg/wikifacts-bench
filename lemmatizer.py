import spacy
import spacy.cli
import subprocess
import sys
from spacy.language import Language
from spacy.tokens import Doc
import pymorphy3
from underthesea import word_tokenize, text_normalize


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

class ChineseLemmatizer:
    def __init__(self, nlp):
        pass
    
    def __call__(self, doc):
        for token in doc:
            token.lemma_ = token.text
        return doc

@Language.factory("chinese_lemmatizer")
def create_chinese_lemmatizer(nlp, name):
    return ChineseLemmatizer(nlp)

class VietnameseLemmatizer:
    def __init__(self, nlp):
        pass

    def __call__(self, doc):
        for token in doc:
            token.lemma_ = token.text

        return doc


@Language.factory("vietnamese_lemmatizer")
def create_vietnamese_lemmatizer(nlp, name):
    return VietnameseLemmatizer(nlp)

class MultilingualLemmatizer:
    def __init__(self, lang: str):
        self.lang = lang
        
        if self.lang == 'vi':
            if word_tokenize is None:
                raise ImportError("underthesea is required for Vietnamese ('vi'). Install via: pip install underthesea")

            self.nlp = spacy.blank('xx')

            def _vi_tokenizer(text):
                norm = text_normalize(text) if text_normalize else text

                segmented = word_tokenize(norm, format='text')  
                words = segmented.split()
               
                return Doc(self.nlp.vocab, words=words)

            self.nlp.tokenizer = _vi_tokenizer
            self.nlp.add_pipe('vietnamese_lemmatizer', last=True)
            return

        model_news = f"{self.lang}_core_news_sm"
        model_web  = f"{self.lang}_core_web_sm"

        try:
            spacy.cli.download(model_news)
            self.nlp = spacy.load(model_news)
        except OSError:
            try:
                spacy.cli.download(model_web)
                self.nlp = spacy.load(model_web)
            except OSError:
                print(f"Neither {model_news} nor {model_web} found. Downloading {model_news}…")
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
        elif self.lang == 'zh':
            self.nlp.add_pipe('chinese_lemmatizer')



    def lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text)
        if self.lang in ('zh', 'vi'):
            lemmas = [tok.lemma_ for tok in doc if any(ch.isalnum() for ch in tok.text)]
        else:
            lemmas = [tok.lemma_ for tok in doc if (tok.is_alpha or tok.is_digit) and len(tok) > 1]
        return ' '.join(lemmas)