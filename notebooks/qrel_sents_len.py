import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from datasets import load_dataset
    return (load_dataset,)


@app.cell
def _(load_dataset):
    ds = load_dataset('kaengreg/wikifacts-sents-qrels', 'qrels')['dev']
    ds
    return (ds,)


@app.cell
def _(ds):
    ids = []
    for entry in ds:
        ids.append(entry['corpus-id'])

    len(ids)
    return (ids,)


@app.cell
def _(ids):
    ids[2]
    return


@app.cell
def _(load_dataset):
    text_ds = load_dataset('kaengreg/wikifacts-sents', 'corpus')
    texts = {}
    for record in text_ds['train']:
        texts[record['_id']] = record['text']

    len(texts)
    return text_ds, texts


@app.cell
def _(ids, texts):
    outputs = [texts[x] for x in ids]
    outputs[:10]
    return (outputs,)


@app.cell
def _(outputs):
    out_lens = [len(x) for x in outputs]
    out_lens[:10]
    return (out_lens,)


@app.cell
def _():
    from collections import Counter

    import matplotlib.pyplot as plt
    import pandas as pd
    return Counter, pd, plt


@app.cell
def _(Counter, out_lens):
    len_data = dict(sorted(Counter(out_lens).items()))
    len_data
    return (len_data,)


@app.cell
def _(len_data, plt):
    x_values = list(len_data.keys())
    y_values = list(len_data.values())

    plt.plot(x_values, y_values)
    plt.xlabel('Sentence Length')
    plt.ylabel('Occurences')
    plt.title('Qrel-sents length distribution')

    plt.show()
    return


@app.cell
def _(len_data, pd):
    data_for_df = {
        'len': list(len_data.keys()),
        'freq': list(len_data.values())
    }

    df = pd.DataFrame(data_for_df)

    # df = pd.DataFrame.from_dict(len_data, orient='index', columns=['Occurences'])
    df
    return (df,)


@app.cell
def _(df):
    import plotly.express as px

    fig = px.line(df, x="len", y="freq", title='Sents length distribution')
    fig.show()
    return


@app.cell
def _(text_ds):
    all_texts = []
    for record_x in text_ds['train']:
        all_texts.append(record_x['text'])

    len(all_texts)
    return (all_texts,)


@app.cell
def _(all_texts):
    all_texts[:10]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
