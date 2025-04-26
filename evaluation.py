from collections import defaultdict
from tqdm import tqdm
import evaluate
from datasets import load_from_disk


def ner_seq_from(chars, ners):
    pred = ["O" for _ in range(len(chars))]

    for ner_index in ners:
        pred[ner_index] = "Sep"

    return pred


def try_fix_last_index(text, indices):
    last_index = indices[-1]

    if last_index >= len(text):
        indices[-1] = last_index - 1


def model_chonky(model_id, **kwargs):
    from src.chonky import ParagraphSplitter

    splitter = ParagraphSplitter(
        model_id=model_id,
        device="cuda",
        **kwargs,
    )

    def predict(x):
        last_char_idx = 0
        pred_char_indices = []

        for chunk in splitter(x):
            last_char_idx += len(chunk)

            pred_char_indices.append(last_char_idx)

        try_fix_last_index(x, pred_char_indices)

        return ner_seq_from(x, pred_char_indices)

    return predict


def model_chonkie_semantic(embedding_model):
    from chonkie import SemanticChunker

    chunker = SemanticChunker(embedding_model=embedding_model)

    def predict(x):
        chunks = chunker(x)

        pred_char_indices = [chunk.end_index for chunk in chunks]
        try_fix_last_index(x, pred_char_indices)

        return ner_seq_from(x, pred_char_indices)

    return predict


def model_chonkie_recursive():
    from chonkie import RecursiveChunker

    chunker = RecursiveChunker()

    def predict(x):
        chunks = chunker(x)

        pred_char_indices = [chunk.end_index for chunk in chunks]
        try_fix_last_index(x, pred_char_indices)

        return ner_seq_from(x, pred_char_indices)

    return predict


def model_sat(model_id, do_paragraph_segmentation):
    from wtpsplit import SaT

    model = SaT(model_id)
    model.to("cuda")

    def predict(x):
        last_char_idx = 0
        pred_char_indices = []

        for sents in model.split(
            x, do_paragraph_segmentation=do_paragraph_segmentation
        ):
            if not do_paragraph_segmentation:
                sents = [sents]

            last_char_idx += sum(map(len, sents))

            pred_char_indices.append(last_char_idx - 1)

        return ner_seq_from(x, pred_char_indices)

    return predict


def model_llama_index_semantic_splitter(embedding_model):
    from llama_index.core.schema import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    splitter = SemanticSplitterNodeParser(
        embed_model=HuggingFaceEmbedding(model_name=embedding_model)
    )

    def predict(x):
        doc = Document(text=x)
        nodes = splitter.get_nodes_from_documents([doc])
        pred_char_indices = [node.end_char_idx - 1 for node in nodes]
        try_fix_last_index(x, pred_char_indices)

        return ner_seq_from(x, pred_char_indices)

    return predict


def model_langchain_semantic_chunker(embedding_model):
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings

    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    splitter = SemanticChunker(hf_embeddings)

    def predict(x):
        docs = splitter.create_documents([x])

        last_char_idx = 0
        pred_char_indices = []

        for doc in docs:
            last_char_idx += len(doc.page_content)

            pred_char_indices.append(last_char_idx + 1)

        try_fix_last_index(x, pred_char_indices)

        return ner_seq_from(x, pred_char_indices)

    return predict


def model_langchain_recursive_chunker():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)

    def predict(x):
        docs = splitter.create_documents([x])

        last_char_idx = 0
        pred_char_indices = []

        for doc in docs:
            last_char_idx += len(doc.page_content)

            pred_char_indices.append(last_char_idx + 1)

        try_fix_last_index(x, pred_char_indices)

        return ner_seq_from(x, pred_char_indices)

    return predict


def make_gt(tokens, ner_tags):
    char_indices = []
    char_it = 0
    for token, tag in zip(tokens, ner_tags):
        char_it += len(token) + 1

        if tag:
            char_indices.append(char_it - 1)

    char_indices[-1] -= 1

    return ner_seq_from(" ".join(tokens), char_indices)


def eval_loop(eval_dataset, models):
    all_outputs = defaultdict(list)
    gts = []

    assert len(eval_dataset["tokens"]) == len(eval_dataset["ner_tags"])

    tokens = eval_dataset["tokens"]
    ner_tags = eval_dataset["ner_tags"]

    outputs_to_assert = []
    for model_name, model in tqdm(models):
        text = " ".join(tokens)

        output = model(text)
        all_outputs[model_name].append(output)
        outputs_to_assert.append(output)

    gt = make_gt(tokens, ner_tags)
    gts.append(gt)

    assert len(set(list(map(len, outputs_to_assert)) + [len(gt)])) == 1

    return all_outputs, gts


def pretty_print_metrics(all_metrics):
    from operator import itemgetter
    from tabulate import tabulate

    headers = ["Model"]
    rows = []

    is_header_set = False
    by_model = itemgetter(0)
    all_metrics = sorted(all_metrics.items(), key=by_model)
    for model_name, for_model in all_metrics:
        by_dataset = itemgetter(0)
        for_model = sorted(for_model, key=by_dataset)

        row = []
        for dataset_name, metrics in for_model:
            if not is_header_set:
                headers.append(dataset_name)

            metric_value = round(metrics["overall_f1"], 2)
            row.append(metric_value)

        row = [model_name] + row
        rows.append(row)
        is_header_set = True

    print(tabulate(rows, headers=headers, tablefmt="github"))


def main():
    dataset_names = [
        "bookcorpus",
        "en_judgements",
        "paul_graham",
        "20_newsgroups",
    ]

    models = [
        (
            "SaT(sat-12l-sm, do_ps=True)",
            model_sat("sat-12l-sm", do_paragraph_segmentation=True),
        ),
        (
            "SaT(sat-12l-sm, do_ps=False)",
            model_sat("sat-12l-sm", do_paragraph_segmentation=False),
        ),
        ("SaT sat-3l do_ps=True", model_sat("sat-3l", do_paragraph_segmentation=True)),
        (
            "SaT(sat-3l, do_ps=False)",
            model_sat("sat-3l", do_paragraph_segmentation=False),
        ),
        (
            "chonkY_distilbert",
            model_chonky(model_id="mirth/chonky_distilbert_uncased_1"),
        ),
        (
            "chonkY_modernbert",
            model_chonky(
                model_id="mirth/chonky_modernbert_base_1",
                _attn_implementation="sdpa",
                reference_compile=False,
            ),
        ),
        (
            "chonkIE SemanticChunker(potion-base-8M)",
            model_chonkie_semantic(embedding_model="minishlab/potion-base-8M"),
        ),
        (
            "chonkIE SemanticChunker(bge-small-en-v1.5)",
            model_chonkie_semantic(embedding_model="BAAI/bge-small-en-v1.5"),
        ),
        ("chonkIE RecursiveChunker", model_chonkie_recursive()),
        (
            "llamaindex SemanticSplitter(bge-small-en-v1.5)",
            model_llama_index_semantic_splitter(
                embedding_model="BAAI/bge-small-en-v1.5"
            ),
        ),
        (
            "langchain SemanticChunker(all-mpnet-base-v2)",
            model_langchain_semantic_chunker(
                embedding_model="sentence-transformers/all-mpnet-base-v2"
            ),
        ),
        (
            "langchain SemanticChunker(bge-small-en-v1.5)",
            model_langchain_semantic_chunker(embedding_model="BAAI/bge-small-en-v1.5"),
        ),
        (
            "langchain SemanticChunker(potion-base-8M)",
            model_langchain_semantic_chunker(
                embedding_model="minishlab/potion-base-8M"
            ),
        ),
        ("langchain RecursiveChar", model_langchain_recursive_chunker()),
    ]

    all_metrics = defaultdict(list)
    for dataset_name in dataset_names:
        eval_dataset = load_from_disk(f"data/{dataset_name}")
        eval_dataset = eval_dataset[:1000000]

        results, gts = eval_loop(eval_dataset, models)

        seqeval = evaluate.load("seqeval")

        for model_name, preds in sorted(results.items(), key=lambda pair: pair[0]):
            metrics = seqeval.compute(predictions=preds, references=gts)
            all_metrics[model_name].append((dataset_name, metrics))

    pretty_print_metrics(all_metrics)


if __name__ == "__main__":
    main()
