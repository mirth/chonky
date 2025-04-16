from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def split_into_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def split_text_into_even_chunks(tokenizer, text):
    ids_plus = tokenizer(
        text, truncation=False, add_special_tokens=False, return_offsets_mapping=True
    )
    token_offset_tups = ids_plus["offset_mapping"]
    offset_tup_chunks = split_into_chunks(
        token_offset_tups, n=tokenizer.model_max_length
    )
    # Map token chunks back to text chunks by using the start index of the first token and the end index of the last token
    chunks = (
        text[offset_tup_chunk[0][0] : offset_tup_chunk[-1][1]]
        for offset_tup_chunk in offset_tup_chunks
    )

    return chunks


def split_into_semantic_chunks(text, ners):
    begin_index = 0

    for i, _c in enumerate(text):
        for ner in ners:
            if i == ner["end"]:
                chunk = text[begin_index : ner["end"]]

                yield chunk

                begin_index = ner["end"]

    yield text[begin_index:]


class TextSplitter:
    def __init__(self, model_id="mirth/chonky_distilbert_uncased_1", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        )
        model.to(device)
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=self.tokenizer,
            device=device,
            aggregation_strategy="simple",
        )

    def __call__(self, text):
        text_chunks = split_text_into_even_chunks(self.tokenizer, text)

        for text_chunk in text_chunks:
            output = self.pipe(text_chunk)

            yield from split_into_semantic_chunks(text_chunk, output)


if __name__ == "__main__":
    with open(
        "../../data/paul_graham_essay_no_new_line/paul_graham_essay_no_new_line.txt"
    ) as file:
        pg = file.read()

    c = TextSplitter(device="cuda")
    for sem_chunk in c(pg):
        print(sem_chunk)
        print("--")
