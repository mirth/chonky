# Chonky

__Chonky__ is a Python library that intelligently segments text into meaningful semantic chunks using a fine-tuned transformer model. This library can be used in the RAG systems.

## Installation

```
pip install chonky
```

Usage:

```
from chonky import TextSplitter

# on the first run it will download the transformer model
splitter = TextSplitter(device="cpu")

text = """Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep. The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights."""

for chunk in splitter(text):
  print(chunk)
  print("--")
```

## Transformer model

[mirth/chonky_distilbert_base_uncased_1](https://huggingface.co/mirth/chonky_distilbert_base_uncased_1)
