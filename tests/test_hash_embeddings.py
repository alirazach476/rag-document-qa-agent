from rag_qa.hash_embeddings import HashEmbeddings


def test_hash_embeddings_deterministic() -> None:
    emb = HashEmbeddings(dimensions=64)
    a = emb.embed_query("project objectives")
    b = emb.embed_query("project objectives")
    assert a == b
    assert len(a) == 64


def test_hash_embeddings_ascii_safe() -> None:
    emb = HashEmbeddings()
    vec = emb.embed_query("Objectives:\u2026 phase one")
    assert all(isinstance(x, float) for x in vec)
