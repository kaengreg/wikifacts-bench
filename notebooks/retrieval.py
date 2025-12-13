import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from fastembed import SparseEmbedding, SparseTextEmbedding
    return (SparseTextEmbedding,)


@app.cell
def _(SparseTextEmbedding):
    SparseTextEmbedding.list_supported_models()
    return


@app.cell
def _(SparseTextEmbedding):
    model_name = "prithivida/Splade_PP_en_v1"

    # This triggers the model download
    model = SparseTextEmbedding(model_name=model_name)
    return (model,)


@app.cell
def _(model):
    model
    return


@app.cell
def _():
    import sys
    sys.path.append("..")
    return


@app.cell
def _():
    from retrieval import RelevantRetriever
    return (RelevantRetriever,)


@app.cell
def _():
    import spacy

    spacy.load('ru_core_news_sm')
    return


@app.cell
def _(RelevantRetriever):
    fact = "Какая столица Франции?"
    article_texts = {
        'c_1': """
    Франция — это страна в Западной Европе. 
    Столица Франции — Париж, который известен своим искусством, модой, гастрономией и культурой.
    Франция известна своим вином и сыром.
    В отличие от США, Франция не имеет национального языка.
    """,
    }

    sparse = RelevantRetriever(mode='sparse')
    return article_texts, fact, sparse


@app.cell
def _(article_texts, fact, sparse):
    frags = sparse.retrieve(fact, article_texts, top_k=2)
    frags
    return


@app.cell
def _(sparse):
    sparse._implementation.tokenizer.nlp.pipeline[-1][1].__dict__
    return


@app.cell
def _():
    from data_loader import load_facts

    queries = load_facts('kaengreg/wikifacts-bench', split='ru_queries')
    corpus = load_facts('kaengreg/wikifacts-bench', split='ru_corpus')
    return corpus, queries


@app.cell
def _(queries):
    queries
    return


@app.cell
def _(corpus):
    corpus
    return


@app.cell
def _():
    import nltk
    nltk.download('stopwords')
    return


if __name__ == "__main__":
    app.run()

