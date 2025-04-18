from operator import itemgetter
from collections import defaultdict
import numpy as np
import pandas as pd
from datasets import Dataset


def dataset_refined_bookcorpus():
    from datasets import load_from_disk

    ds = load_from_disk("data/refined-bookcorpus-dataset_hf_test")

    all_paragraphs = []
    for tokens, ner_tags in zip(ds["tokens"], ds["ner_tags"]):
        prev_tag_index = 0

        for tag_index in np.nonzero(ner_tags)[0]:
            tag_index += 1
            para = " ".join(tokens[prev_tag_index:tag_index])
            all_paragraphs.append(para)

            prev_tag_index = tag_index

    return all_paragraphs


def dataset_MultiLegalSBD_en_judgements():
    filenames = [
        "data/MultiLegalSBD/data/en_judgements_0.jsonl",
        "data/MultiLegalSBD/data/en_judgements_1.jsonl",
        "data/MultiLegalSBD/data/en_judgements_2.jsonl",
        "data/MultiLegalSBD/data/en_judgements_3.jsonl",
    ]

    dfs = []
    for filename in filenames:
        dfs.append(pd.read_json(filename, lines=True))

    df = pd.concat(dfs)

    all_paragraphs = []

    for text in df.text.values:
        paragraphs = list(filter(len, text.split("\r\n")))

        all_paragraphs.extend(paragraphs)

    return all_paragraphs


def hf_dataset_with_paragraphs(dataset_id, split):
    from datasets import load_dataset

    ds = load_dataset(dataset_id)

    all_paragraphs = []

    for text in ds[split]["text"]:
        paragraphs = list(filter(len, text.split("\n")))

        all_paragraphs.extend(paragraphs)

    return all_paragraphs


def to_dict(paragraphs):
    tokens = []
    ner_tags = []

    for para in paragraphs:
        for token in para.split(" "):
            tokens.append(token)
            ner_tags.append(0)

        ner_tags[-1] = 1

    assert len(tokens) == len(ner_tags)

    return {"tokens": tokens, "ner_tags": ner_tags}


def main():
    raw_datasets = [
        ("bookcorpus", dataset_refined_bookcorpus()),
        ("en_judgements", dataset_MultiLegalSBD_en_judgements()),
        (
            "paul_graham",
            hf_dataset_with_paragraphs(
                dataset_id="sgoel9/paul_graham_essays", split="train"
            ),
        ),
        (
            "20_newsgroups",
            hf_dataset_with_paragraphs(dataset_id="SetFit/20_newsgroups", split="test"),
        ),
    ]

    for dataset_name, paragraphs in raw_datasets:
        dataset = Dataset.from_dict(to_dict(paragraphs))
        dataset.save_to_disk(f"data/{dataset_name}")


if __name__ == "__main__":
    main()
