from pypdf import PdfReader

reader = PdfReader("data/raw/patents_act_1970.pdf")
keys = ["What are not inventions", "Persons entitled to apply", "Inventions relating to atomic", "Definitions"]

for n, page in enumerate(reader.pages[:14], start=1):
    for line in (page.extract_text() or "").split("\n"):
        if any(k in line for k in keys):
            print(n, repr(line[:140]))