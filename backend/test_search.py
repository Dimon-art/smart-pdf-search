from services.pdf_parser import extract_pages_with_bbox, clean_text
from services.embedder import chunk_text, build_embeddings, build_faiss_index, search, save_cache, load_cache
import hashlib
from sentence_transformers import SentenceTransformer


def main():
    """Parse sample.pdf (or load cache), then run semantic search queries."""
    pdf_hash = hashlib.md5(open('../sample.pdf', 'rb').read()).hexdigest()
    print(f'PDF hash: {pdf_hash[:8]}...')

    cached = load_cache(pdf_hash)
    if cached is not None:
        chunks, embeddings, index = cached
        print('Загружено из кэша')
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    else:
        pages = extract_pages_with_bbox('../sample.pdf')
        full_text = clean_text(' '.join(b['text'] for p in pages for b in p['blocks']))
        print(f'Длина текста: {len(full_text)}')
        chunks = chunk_text(full_text, chunk_size=500, overlap=100)
        print(f'Чанков: {len(chunks)}')
        embeddings, model = build_embeddings(chunks)
        print(f'Размер эмбеддингов: {embeddings.shape}')
        index = build_faiss_index(embeddings)
        print(f'FAISS-индекс: {index.ntotal} векторов')
        save_cache(pdf_hash, chunks, embeddings, index)
        print('Кэш сохранён')

    queries = [
        'Russian interference',
        'вмешательство России',
        'Trump campaign',
        'предвыборная кампания Трампа',
        'obstruction of justice',
        'воспрепятствование правосудию',
    ]
    print('\nСравнение запросов:')
    for q in queries:
        results = search(index, model, q, chunks, top_k=1)
        if results:
            r = results[0]
            print(f'\n{q!r}: score={r["score"]:.4f}')
            print(f'  Найден чанк: {r["chunk"][:120]}...')
        else:
            print(f'\n{q!r}: нет результатов')


if __name__ == '__main__':
    main()
