from .markup_remover import MarkupRemover


def test_markdown():
    text = """
# Header 1
## Header 2
### Header 3
#### Header 4
##### Header 5
###### Header 6

- List item 1
- List item 2
- List item 3

[Link](https://example.com)
"""
    stripper = MarkupRemover()
    result = stripper(text)

    assert (
        result
        == "Header 1 Header 2 Header 3 Header 4 Header 5 Header 6 List item 1 List item 2 List item 3 Link"
    )


def test_html():
    text = """
    <html>
        <head>
            <title>Title</title>
        </head>
        <body>
            <h1>Header 1</h1>
            <p>Paragraph 1</p>
            <a href="https://example.com">Link</a>
        </body>
    </html>
    """
    stripper = MarkupRemover()
    result = stripper(text)

    assert result == "Title Header 1 Paragraph 1 Link"


def test_xml():
    text = """
    <note>
        <to>Tove</to>
        <from>Jani</from>
        <heading>Reminder</heading>
        <body>Don't forget me this weekend!</body>
    </note>
    """
    stripper = MarkupRemover()
    result = stripper(text)

    assert result == "Tove Jani Reminder Don't forget me this weekend!"


def test_other():
    text = "This is a plain text without any markup."
    stripper = MarkupRemover()
    result = stripper(text)

    assert result == text


def test_empty():
    text = ""
    stripper = MarkupRemover()
    result = stripper(text)

    assert result == ""


def test_whitespace():
    text = "    \n\t Text \r  "
    stripper = MarkupRemover()
    result = stripper(text)

    assert result == "Text"
