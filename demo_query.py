from pipeline import RAGPipeline

query = input("Enter your question: ")

p = RAGPipeline(
    pdf_dir="./data/sample_pdfs",
    index_path="./index"
)

r = p.query(query)
if r["confidence"] < 0.65:
    print("\nLow confidence result.")
    print("No sufficiently relevant evidence found in the provided documents.")
    exit()
print("\nConfidence:", r["confidence"])

print("\nAnswer Preview:\n")
print(r["answer"][:500])

print("\nSources:")
for s in r["sources"]:
    print(f'- {s["document"]} | page {s["page"]}')