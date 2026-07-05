from __future__ import annotations

import pytest

import bicardinal
from bicardinal import Bicardinal
from bicardinal import Config

class _Usage:
    def __init__(self, i=11, o=7):
        self.input_tokens = i
        self.output_tokens = o


class _Resp:
    def __init__(self, text: str):
        self.output_text = text
        self.usage = _Usage()


class _Responses:
    def create(self, *, model, instructions=None, input, **kw) -> _Resp:
        text = input[0] if isinstance(input, list) else str(input)
        return _Resp(text.removeprefix("CONTEXT: ").strip())


class _Transcriptions:
    def create(self, *, model, file, **kw):
        return type("T", (), {"text": "fake transcript"})()


class FakeOpenAI:
    def __init__(self, *a, **kw):
        self.responses = _Responses()
        self.audio = type("A", (), {"transcriptions": _Transcriptions()})()


class _Ocr:
    def process(self, *, model, document, **kw):
        page = type("P", (), {"markdown": "fake ocr page"})()
        return type("R", (), {"pages": [page]})()


class FakeMistral:
    def __init__(self, *a, **kw):
        self.ocr = _Ocr()


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(bicardinal, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(bicardinal, "Mistral", FakeMistral)
    return Bicardinal(
        tmp_path / "data",
        config=Config(chunk_size=128, overlap=0.1),
        openai_api_key="test",
        mistral_api_key="test",
    )


def test_ingest_and_search(store):
    col = store.create("demo")
    col.init("build")

    col.ingest(
        "cats.txt", b"Cats are small domesticated felines that purr and chase mice."
    )
    col.ingest(
        "finance.txt", b"Quarterly revenue grew as the company cut operating costs."
    )
    assert col.finalize() == {}

    st = col.status()
    assert st.n_files == 2
    assert st.n_chunks >= 2

    hits = col.search("a pet animal that meows", k=2)
    assert hits, "expected search results"
    assert hits[0].filename == "cats.txt"

    col.close()


def _make_corpus() -> dict[str, bytes]:
    topics = {
        "astro": (
            "The telescope observed a distant spiral galaxy and a glowing nebula. "
            "Astronomers tracked the orbit of the planet around its bright star, "
            "measuring stellar parallax and the redshift of faraway quasars. "
        ),
        "cook": (
            "The recipe simmered garlic and onions in olive oil before roasting. "
            "Season the sauce with basil, fold in the dough, and bake until golden. "
            "A pinch of salt balances the sweetness of the caramelized flavor. "
        ),
        "finance": (
            "Quarterly revenue rose after the company cut operating expenses. "
            "The portfolio paid a dividend while invoices and receivables cleared. "
            "Analysts revised earnings guidance ahead of the shareholder meeting. "
        ),
    }
    corpus: dict[str, bytes] = {}
    for topic, body in topics.items():
        for i in range(12):
            text = f"Document {i} about {topic}. " + (body * 6)
            corpus[f"{topic}_{i:02d}.txt"] = text.encode()
    return corpus


def test_large_corpus_ingest_search_and_delete(store):
    corpus = _make_corpus()
    col = store.create("big")
    col.init("build")

    total_chunks = 0
    for name, data in corpus.items():
        res = col.ingest(name, data)
        assert res.errors == []
        total_chunks += res.n_chunks
    assert col.finalize() == {}

    st = col.status()
    assert st.n_files == len(corpus) == 36
    assert st.n_chunks == total_chunks
    assert st.n_chunks > st.n_files

    queries = {
        "astro": "stargazing with a powerful lens to study the cosmos",
        "cook": "preparing a tasty meal in the kitchen with herbs",
        "finance": "company profits, earnings and stock dividends",
    }
    for topic, query in queries.items():
        hits = col.search(query, k=5)
        assert hits, f"no hits for {topic!r}"
        assert hits[0].filename.startswith(topic), f"{query!r} -> {hits[0].filename}"
        same = sum(h.filename.startswith(topic) for h in hits)
        assert same >= 3

    files = col.most_similar_files(queries["cook"], k=3)
    assert files and files[0].filename.startswith("cook")
    assert files[0].best_chunk.filename == files[0].filename

    target = files[0].filename
    scoped = col.search_in_file("baking and seasoning", target, k=5)
    assert scoped and all(h.filename == target for h in scoped)

    col.delete(target)
    st2 = col.status()
    assert st2.n_files == st.n_files - 1
    assert target not in st2.filenames
    assert all(h.filename != target for h in col.search(queries["cook"], k=10))

    col.close()
    reopened = store.open("big")
    assert reopened.status().n_files == st2.n_files
    hits = reopened.search(queries["astro"], k=3)
    assert hits and hits[0].filename.startswith("astro")
    reopened.close()


# --- weird / edge-case scenarios


def test_empty_file_is_rejected(store):
    col = store.create("edge_empty")
    col.init("build")
    with pytest.raises(bicardinal.EmptyFile):
        col.ingest("nothing.txt", b"")


def test_unsupported_file_type_is_rejected(store):
    col = store.create("edge_unsupported")
    col.init("build")
    zip_bytes = b"PK\x03\x04" + b"\x00" * 64
    with pytest.raises(bicardinal.UnsupportedFileType):
        col.ingest("archive.zip", zip_bytes)


def test_duplicate_filename_is_rejected(store):
    col = store.create("edge_dupe")
    col.init("build")
    col.ingest("a.txt", b"first ingestion of this document about otters")
    with pytest.raises(bicardinal.DuplicateDocument):
        col.ingest("a.txt", b"completely different text, same filename")


def test_queries_on_empty_collection(store):
    col = store.create("edge_empty_query")
    col.init("build")
    col.finalize()

    assert col.search("anything") == []
    assert col.most_similar_files("anything") == []


def test_operations_on_missing_file_raise(store):
    col = store.create("edge_missing")
    col.init("build")
    col.ingest("real.txt", b"a real document about lighthouses and the sea")
    col.finalize()
    with pytest.raises(bicardinal.DocumentNotFound):
        col.search_in_file("query", "ghost.txt")
    with pytest.raises(bicardinal.DocumentNotFound):
        col.delete("ghost.txt")


def test_whitespace_only_doc_registers_with_zero_chunks(store):
    col = store.create("edge_blank")
    col.init("build")
    res_blank = col.ingest("blank.txt", b"   \n\t  \r\n   ")
    res_real = col.ingest("real.txt", b"penguins huddle together on the antarctic ice")
    col.finalize()

    assert res_blank.n_chunks == 0
    assert res_real.n_chunks > 0

    st = col.status()
    assert st.n_files == 2
    assert "blank.txt" in st.filenames
    assert st.n_chunks == res_real.n_chunks

    hits = col.search("birds on ice", k=10)
    assert hits and all(h.filename != "blank.txt" for h in hits)


def test_unicode_and_emoji_roundtrip(store):
    col = store.create("edge_unicode")
    col.init("build")
    body = "Café résumé naïve façade — quantum entanglement of qubits 🧪⚛️ in a lab."
    col.ingest("unicode.txt", body.encode("utf-8"))
    col.finalize()
    hits = col.search("physics experiment with quantum particles", k=1)
    assert hits
    assert "🧪" in hits[0].raw_text and "Café" in hits[0].raw_text


def test_k_larger_than_corpus_does_not_crash(store):
    col = store.create("edge_bigk")
    col.init("build")
    for i in range(3):
        col.ingest(f"doc_{i}.txt", f"short note number {i} about volcanoes".encode())
    col.finalize()
    total = col.status().n_chunks
    hits = col.search("geology and eruptions", k=1000)
    assert 0 < len(hits) <= total


def test_identical_content_distinct_filenames(store):
    col = store.create("edge_identical")
    col.init("build")
    body = b"the exact same sentence about coral reefs and marine biology"
    col.ingest("copy_a.txt", body)
    col.ingest("copy_b.txt", body)
    col.finalize()

    assert col.status().n_files == 2
    names = {h.filename for h in col.search("coral reef ecosystems", k=10)}
    assert {"copy_a.txt", "copy_b.txt"} <= names
    files = {f.filename for f in col.most_similar_files("coral reef ecosystems", k=5)}
    assert files == {"copy_a.txt", "copy_b.txt"}


def test_summarizer_failures_fall_back_to_raw_text(tmp_path, monkeypatch):
    import threading

    class FlakyResponses:
        def __init__(self):
            self._lock = threading.Lock()
            self._calls = 0

        def create(self, *, input, **kw):
            with self._lock:
                self._calls += 1
                first = self._calls == 1
            if first:
                raise RuntimeError("transient model error")
            text = input[0] if isinstance(input, list) else str(input)
            return _Resp(text.removeprefix("CONTEXT: ").strip())

    class FlakyOpenAI:
        def __init__(self, *a, **kw):
            self.responses = FlakyResponses()
            self.audio = FakeOpenAI().audio

    monkeypatch.setattr(bicardinal, "OpenAI", FlakyOpenAI)
    monkeypatch.setattr(bicardinal, "Mistral", FakeMistral)

    store = Bicardinal(
        tmp_path / "data",
        config=Config(chunk_size=32, overlap=0.1),
        openai_api_key="test",
        mistral_api_key="test",
    )
    col = store.create("flaky")
    col.init("build")

    body = " ".join(
        f"Wetland fact {i}: migratory birds depend on coastal marshes."
        for i in range(30)
    )
    res = col.ingest("birds.txt", body.encode())
    assert col.finalize() == {}
    assert res.n_chunks > 1
    assert res.errors
    assert any("transient model error" in e for e in res.errors)

    hits = col.search("birds and coastal marshes", k=3)
    assert hits and hits[0].filename == "birds.txt"
    col.close()

# --- reading chunks


def test_read_chunks_returns_all_in_order(store):
    col = store.create("read_all")
    col.init("build")
    body = " ".join(
        f"Sentence {i} about coral reefs, marine biology, and ocean currents."
        for i in range(60)
    )
    res = col.ingest("reef.txt", body.encode())
    col.finalize()
    assert res.n_chunks > 1, "need multiple chunks to test ordering"

    chunks = col.read_chunks("reef.txt")
    assert len(chunks) == res.n_chunks
    assert [c.chunk_index for c in chunks] == list(range(res.n_chunks))
    assert all(c.filename == "reef.txt" for c in chunks)
    assert all(c.raw_text for c in chunks)
    assert all(c.chunk_id.startswith("ch_") for c in chunks)

    col.close()


def test_read_chunks_pagination(store):
    col = store.create("read_page")
    col.init("build")
    body = " ".join(
        f"Fact {i}: migratory birds depend on coastal wetlands and marshes."
        for i in range(60)
    )
    res = col.ingest("birds.txt", body.encode())
    col.finalize()
    assert res.n_chunks >= 4, "need enough chunks to slice"

    everything = col.read_chunks("birds.txt")

    tail = col.read_chunks("birds.txt", start=2)
    assert [c.chunk_index for c in tail] == [c.chunk_index for c in everything[2:]]

    head = col.read_chunks("birds.txt", limit=2)
    assert [c.chunk_index for c in head] == [0, 1]

    window = col.read_chunks("birds.txt", start=1, limit=2)
    assert [c.chunk_index for c in window] == [1, 2]

    assert col.read_chunks("birds.txt", start=10_000) == []
    assert len(col.read_chunks("birds.txt", start=2, limit=10_000)) == res.n_chunks - 2

    col.close()


def test_read_chunks_missing_file_raises(store):
    col = store.create("read_missing")
    col.init("build")
    col.ingest("real.txt", b"a real document about lighthouses and the sea")
    col.finalize()
    with pytest.raises(bicardinal.DocumentNotFound):
        col.read_chunks("ghost.txt")


def test_read_chunks_zero_chunk_file_is_empty(store):
    col = store.create("read_blank")
    col.init("build")
    col.ingest("blank.txt", b"   \n\t  \r\n   ")
    col.finalize()
    assert col.read_chunks("blank.txt") == []


def test_read_chunks_matches_search_hit_content(store):
    col = store.create("read_vs_search")
    col.init("build")
    col.ingest("reef.txt", b"the exact same sentence about coral reefs and marine biology")
    col.finalize()

    chunks = col.read_chunks("reef.txt")
    assert chunks
    hits = col.search_in_file("coral reefs", "reef.txt", k=len(chunks))
    assert hits
    by_id = {c.chunk_id: c for c in chunks}
    for h in hits:
        assert h.chunk_id in by_id
        c = by_id[h.chunk_id]
        assert c.raw_text == h.raw_text
        assert c.description == h.description
        assert c.chunk_index == h.chunk_index

    col.close()


# dual encoding


def _dual_store(tmp_path, monkeypatch, **overrides):
    monkeypatch.setattr(bicardinal, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(bicardinal, "Mistral", FakeMistral)
    config = Config(chunk_size=128, overlap=0.1, dual_encoding=True, **overrides)
    return Bicardinal(
        tmp_path / "data",
        config=config,
        openai_api_key="test",
        mistral_api_key="test",
    )


def test_dual_encoding_ingest_search_and_reopen(tmp_path, monkeypatch):
    store = _dual_store(tmp_path, monkeypatch)
    col = store.create("dual")
    col.init("build")
    col.ingest(
        "cats.txt", b"Cats are small domesticated felines that purr and chase mice."
    )
    col.ingest(
        "finance.txt", b"Quarterly revenue grew as the company cut operating costs."
    )
    assert col.finalize() == {}

    assert col._index._vector_dim == col._embedder.dim * 2

    hits = col.search("a pet animal that meows", k=2)
    assert hits and hits[0].filename == "cats.txt"

    for w in (0.0, 0.5, 1.0):
        assert col.search("a pet animal that meows", k=1, fusion_weight=w)

    col.close()

    reopened = _dual_store(tmp_path, monkeypatch).open("dual")
    assert reopened._index._vector_dim == reopened._embedder.dim * 2
    hits = reopened.search("quarterly earnings and revenue", k=1)
    assert hits and hits[0].filename == "finance.txt"
    reopened.close()


def test_dual_encoding_fusion_weight_routes_between_raw_and_description(
    tmp_path, monkeypatch
):
    CAT = "A small domesticated feline pet that purrs and chases mice."
    FIN = "Quarterly corporate revenue, earnings, dividends and operating costs."

    class SwapResponses:
        def create(self, *, model, instructions=None, input, **kw):
            text = input[0] if isinstance(input, list) else str(input)
            raw = text.removeprefix("CONTEXT: ").strip()
            desc = FIN if ("feline" in raw or "purr" in raw) else CAT
            return _Resp(desc)  # describe each chunk as its opposite topic

    class SwapOpenAI:
        def __init__(self, *a, **kw):
            self.responses = SwapResponses()
            self.audio = FakeOpenAI().audio

    monkeypatch.setattr(bicardinal, "OpenAI", SwapOpenAI)
    monkeypatch.setattr(bicardinal, "Mistral", FakeMistral)
    store = Bicardinal(
        tmp_path / "data",
        config=Config(chunk_size=128, overlap=0.1, dual_encoding=True),
        openai_api_key="test",
        mistral_api_key="test",
    )
    col = store.create("swap")
    col.init("build")
    col.ingest("petraw.txt", CAT.encode())
    col.ingest("finraw.txt", FIN.encode())
    assert col.finalize() == {}

    q = "a pet animal that meows"

    raw_only = col.search(q, k=2, fusion_weight=0.0)
    assert raw_only and raw_only[0].filename == "petraw.txt"

    desc_only = col.search(q, k=2, fusion_weight=1.0)
    assert desc_only and desc_only[0].filename == "finraw.txt"

    assert raw_only[0].filename != desc_only[0].filename
    col.close()