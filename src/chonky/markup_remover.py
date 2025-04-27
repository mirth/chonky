import warnings
import re

try:
    import markdown
    from magika import Magika
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "Required libraries are not installed. Please install 'markdown', 'magika', and 'beautifulsoup4'."
    )


class MarkupRemover:
    def __init__(self):
        self.magic = Magika()

    def __call__(self, text):
        res = self.magic.identify_bytes(bytes(text, encoding="raw_unicode_escape"))
        label = res.output.label

        if label == "markdown":
            html = markdown.markdown(text)
            text = self._html_to_text(html, features="lxml")
        elif label == "html":
            text = self._html_to_text(text, features="lxml")
        elif label in ("xml", "asp"):
            text = self._html_to_text(text, features="xml")
        elif label == "txt":
            pass
        else:
            warnings.warn(
                f"MarkupRemover: Unknown markup type '{label}' detected. "
                "Returning the original text."
            )

        text = text.replace("\n", " ")
        text = text.replace("\r", " ")
        text = text.replace("\t", " ")
        text = re.sub(r"\s\s+", " ", text)
        text = text.strip()

        return text

    def _html_to_text(self, text, features):
        soup = BeautifulSoup(text, features=features)
        return soup.get_text().strip()
