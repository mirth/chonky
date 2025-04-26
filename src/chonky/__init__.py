from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def split_into_semantic_chunks(text, ners):
    begin_index = 0

    for ner in ners:
        chunk = text[begin_index : ner["end"]]
        yield chunk
        begin_index = ner["end"]

    yield text[begin_index:]


class ParagraphSplitter:
    def __init__(
        self, model_id="mirth/chonky_distilbert_uncased_1", device="cpu", **model_kwargs
    ):
        if model_id in (
            "mirth/chonky_modernbert_base_1",
            "mirth/chonky_modernbert_large_1",
        ):
            tokenizer_kwargs = {"model_max_length": 1024}
        else:
            tokenizer_kwargs = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        id2label = {
            0: "O",
            1: "separator",
        }
        label2id = {
            "O": 0,
            "separator": 1,
        }

        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            **model_kwargs,
        )
        model.to(device)
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=self.tokenizer,
            device=device,
            aggregation_strategy="simple",
            stride=self.tokenizer.model_max_length // 2,
        )

    def __call__(self, text):
        output = self.pipe(text)

        yield from split_into_semantic_chunks(text, output)


if __name__ == "__main__":
    with open(
        "../../data/paul_graham_essay_no_new_line/paul_graham_essay_no_new_line.txt"
    ) as file:
        pg = file.read()

    c = ParagraphSplitter(device="cuda")
    for sem_chunk in c(pg):
        print(sem_chunk)
        print("--")
