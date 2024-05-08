import embeddings
import pprint as pp

def text_query(client, index_name, query):
    query_emb = embeddings.encode(query)

    query_denc = {
    'size': 5,
    '_source': ['title', 'description', 'ingredients'],
    "query": {
            "knn": {
            "title_embedding": {
                "vector": query_emb[0].numpy(),
                "k": 2
            }
            }
        }
    }

    response = client.search(
        body = query_denc,
        index = index_name
    )

    print('\nSearch results:')
    pp.pprint(response)

