# Chonky

__Chonky__ is a Python library that intelligently segments text into meaningful semantic chunks using a fine-tuned transformer model. This library can be used in the RAG systems.

## Installation

```
pip install chonky
```

Usage:

```python
from chonky import TextSplitter

# on the first run it will download the transformer model
splitter = TextSplitter(device="cpu")

# Or you can select the model
# splitter = TextSplitter(
#  model_id="mirth/chonky_modernbert_base_1",
#  device="cpu"
# )

text = """Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep. The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights."""

for chunk in splitter(text):
  print(chunk)
  print("--")

# Output

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.
--
The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it.
--
 It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
--
```

## Supported models

| Model ID    | F1 | Precision | Recall | Accuracy | Seq Length |
| -------- | ------- | ------- | ------- | ------- | ---------- |
| [mirth/chonky_modernbert_base_1](https://huggingface.co/mirth/chonky_modernbert_base_1) | 0.79 | 0.83 | 0.75 | 0.99 | 1024 |
| [mirth/chonky_distilbert_base_uncased_1](https://huggingface.co/mirth/chonky_distilbert_base_uncased_1)  | 0.7 | 0.79 | 0.63 | 0.99 | 512 |

Metrics above are token based.

## Benchmarks

The following values are character based F1 scores computed on first 1M characters of each datasets (due to performance reasons).

The `bookcorpus` dataset here is basically Chonky validation set so may be it's a bit unfair to list it here.

The `do_ps` fragment for SaT models here is `do_paragraph_segmentation` flag.

| Model                                          |   20_newsgroups       |      bookcorpus     |    en_judgements  |   paul_graham    |
|------------------------------------------------|-----------------------|---------------------|-------------------|------------------|
| chonkY_modernbert                              |            0.15       |         __0.72__ ❗ |        __0.08__ ❗ |      __0.63__ ❗ |
| chonkY_distilbert                              |            0.15       |         0.69        |            0.05   |          0.52    |
| SaT(sat-12l-sm, do_ps=False)                   |            0.31       |         0.33        |            0.03   |          0.43    |
| SaT(sat-12l-sm, do_ps=True)                    |            0.3        |         0.33        |            0.06   |          0.42    |
| SaT(sat-3l, do_ps=False)                       |            __0.34__ ❗ |         0.28       |            0.03   |          0.42    |
| SaT(sat-3l, do_ps=True)                        |            0.15       |         0.09        |            0.07   |          0.41    |
| chonkIE SemanticChunker(bge-small-en-v1.5)     |            0.06       |         0.21        |            0.01   |          0.12    |
| chonkIE SemanticChunker(potion-base-8M)        |            0.08       |         0.19        |            0.01   |          0.15    |
| chonkIE RecursiveChunker                       |            0.02       |         0.07        |            0.01   |          0.05    |
| langchain SemanticChunker(all-mpnet-base-v2)   |            0          |         0           |            0      |          0       |
| langchain SemanticChunker(bge-small-en-v1.5)   |            0          |         0           |            0      |          0       |
| langchain SemanticChunker(potion-base-8M)      |            0          |         0           |            0      |          0       |
| langchain RecursiveChar                        |            0          |         0           |            0      |          0       |
| llamaindex SemanticSplitter(bge-small-en-v1.5) |            0.02       |         0.06        |            0      |          0.06    |
