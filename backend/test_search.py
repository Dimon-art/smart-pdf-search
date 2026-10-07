from services.pdf_parser import extract_pages_with_bbox, clean_text
from services.embedder import chunk_text, build_embeddings, build_faiss_index, search


def main():
    # 1. Извлекаем текст с координатами
    pages = extract_pages_with_bbox('../sample.pdf')
    full_text = clean_text(' '.join(b['text'] for p in pages for b in p['blocks']))
    print(f'Длина текста: {len(full_text)}')

    # 2. Чанкинг
    chunks = chunk_text(full_text, chunk_size=500, overlap=100)
    print(f'Чанков: {len(chunks)}')

    # 3. Эмбеддинги и индекс
    embeddings, model = build_embeddings(chunks)
    print(f'Размер эмбеддингов: {embeddings.shape}')

    index = build_faiss_index(embeddings)
    print(f'FAISS-индекс: {index.ntotal} векторов')

    # 4. Сравнение запросов
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

