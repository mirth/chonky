# Chonky

__Chonky__ is a Python library that intelligently segments text into meaningful semantic chunks using a fine-tuned transformer model. This library can be used in the RAG systems.

## Installation

```
pip install chonky
```

## Usage:

```python
from chonky import ParagraphSplitter

# on the first run it will download the transformer model
splitter = ParagraphSplitter(device="cpu")

# Or you can select the model
# splitter = ParagraphSplitter(
#  model_id="mirth/chonky_modernbert_base_1",
#  device="cpu"
# )

text = (
    "Before college the two main things I worked on, outside of school, were writing and programming. "
    "I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. "
    "My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep. "
    "The first programs I tried writing were on the IBM 1401 that our school district used for what was then called 'data processing.' "
    "This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, "
    "and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — "
    "CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights."
)

for chunk in splitter(text):
  print(chunk)
  print("--")
```

### Sample Output
```
Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.
--
The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it.
--
 It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.
--
```

The usage pattern is the following: strip all the markup tags to produce pure text and feed this text into the splitter. For this purpose there is helper class `MarkupRemover` (it automatically detects the content format):


```python
from chonky.markup_remover import MarkupRemover
from chonky import ParagraphSplitter

remover = MarkupRemover()
splitter = ParagraphSplitter()

text = remover("# Header 1 ...")
splitter(text)
```

Supported formats: `markdown`, `xml`, `html`.


## Supported models

| Model ID                                                                                                       | Seq Length | Number of Params | Multilingual |
| ---------------------------------------------------------------------------------------------------------------| ---------- | ---------------- | ------------ |
| [mirth/chonky_modernbert_large_1](https://huggingface.co/mirth/chonky_modernbert_large_1)                      | 1024       | 396M             | ❌           |
| [mirth/chonky_modernbert_base_1](https://huggingface.co/mirth/chonky_modernbert_base_1)                        | 1024       | 150M             | ❌           |
| [mirth/chonky_mmbert_small_multilingual_1](https://huggingface.co/mirth/chonky_mmbert_small_multilingual_1) 🆕 | 1024       | 140M             | ✅           |
| [mirth/chonky_distilbert_base_uncased_1](https://huggingface.co/mirth/chonky_distilbert_base_uncased_1)        | 512        | 66.4M            | ❌           |


## Benchmarks

The following values are token based F1 scores computed on first 1M tokens of each datasets (due to performance reasons).

### Various English datasets:

The `do_ps` fragment for SaT models here is `do_paragraph_segmentation` flag.

| Model                                          |          bookcorpus   |    en_judgements    |   paul_graham    | 20_newsgroups    |
|------------------------------------------------|-----------------------|---------------------|------------------|------------------|
| chonkY_modernbert_large_1                      |           __0.79__ ❗  |       __0.29__ ❗   |    __0.69__ ❗   | 0.17             |
| chonkY_modernbert_base_1                       |           0.72        |            0.08     |          0.63    | 0.15             |
| chonkY_mmbert_small_multilingual_1 🆕          |           0.72        |            0.2      |          0.56    | 0.13             |
| chonkY_distilbert_base_uncased_1               |           0.69        |            0.05     |          0.52    | 0.15             |
| SaT(sat-12l-sm, do_ps=False)                   |           0.33        |            0.03     |          0.43    | 0.31             |
| SaT(sat-12l-sm, do_ps=True)                    |           0.33        |            0.06     |          0.42    | 0.3              |
| SaT(sat-3l, do_ps=False)                       |           0.28        |            0.03     |          0.42    |  __0.34__ ❗      |
| SaT(sat-3l, do_ps=True)                        |           0.09        |            0.07     |          0.41    | 0.15             |
| chonkIE SemanticChunker(bge-small-en-v1.5)     |           0.21        |            0.01     |          0.12    | 0.06             |
| chonkIE SemanticChunker(potion-base-8M)        |           0.19        |            0.01     |          0.15    | 0.08             |
| chonkIE RecursiveChunker                       |           0.07        |            0.01     |          0.05    | 0.02             |
| langchain SemanticChunker(all-mpnet-base-v2)   |           0           |            0        |          0       | 0                |
| langchain SemanticChunker(bge-small-en-v1.5)   |           0           |            0        |          0       | 0                |
| langchain SemanticChunker(potion-base-8M)      |           0           |            0        |          0       | 0                |
| langchain RecursiveChar                        |           0           |            0        |          0       | 0                |
| llamaindex SemanticSplitter(bge-small-en-v1.5) |           0.06        |            0        |          0.06    | 0.02             |

### Project Gutenberg validation:

| Model                              |   de        |   en       |   es        |   fr        |   it       |   nl         |   pl        |   pt       |   ru        |   sv        |   zh       |
|------------------------------------|-------------|------------|-------------|-------------|------------|--------------|-------------|------------|-------------|-------------|------------|
| chonky_mmbert_small_multi_1 🆕     | __0.88__ ❗ | __0.78__ ❗ | __0.91__ ❗ | __0.93__ ❗ | __0.86__ ❗ | __0.81__  ❗ | __0.81__ ❗ | __0.88__ ❗ | __0.97__ ❗ | __0.91__ ❗  | 0.11       |
| chonky_modernbert_large_1          | 0.53       | 0.43        | 0.48        | 0.51        | 0.56       | 0.21         | 0.65        | 0.53       | 0.87        | 0.51        | __0.33__ ❗ |
| chonky_modernbert_base_1           | 0.42       | 0.38        | 0.34        | 0.4         | 0.33       | 0.22         | 0.41        | 0.35       | 0.27        | 0.31        | 0.26        |
| chonky_distilbert_base_uncased_1   | 0.19       |  0.3        | 0.17        |  0.2        | 0.18       | 0.04         | 0.27        | 0.21       | 0.22        | 0.19        | 0.15       |
| Number of val tokens               | 1M         | 1M          | 1M          | 1M          | 1M         | 1M           | 38K         | 1M         | 24K         | 1M          | 132K       |
