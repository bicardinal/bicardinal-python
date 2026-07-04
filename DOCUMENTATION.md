# bicardinal

bicardinal is a multimodal retrieval engine. You give it files (PDF, Word
documents, images, audio, or plain text). It extracts their contents,
summarizes each piece, turns the summaries into vectors, and stores them in a
searchable index. You then run text queries and get back the most relevant
pieces of content across every file you added.

The library handles the full path from raw bytes to search results, so you do
not write extraction, chunking, embedding, or indexing code yourself.

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Provider keys](#provider-keys)
- [Supported file types](#supported-file-types)
- [Core concepts](#core-concepts)
- [Quick start](#quick-start)
- [Use cases](#use-cases)
- [API reference](#api-reference)
  - [Bicardinal](#bicardinal-1)
  - [Collection](#collection)
  - [Config](#config)
  - [Result types](#result-types)
  - [Exceptions](#exceptions)
- [Behavior notes](#behavior-notes)

## How it works

When you add a file, bicardinal runs these steps:

1. Detect the file type from its bytes.
2. Extract text. PDFs go through optical character recognition, images go
   through a vision model that writes a description and transcribes any text,
   audio goes through speech transcription, and Word and plain text files are
   read directly.
3. Split the extracted text into overlapping chunks sized by token count.
4. Generate a short description (summary) for each chunk.
5. Create a vector for each chunk by embedding its description.
6. Store the vectors in a vector index, and store the original chunk text and
   description alongside.

One detail matters when reading results: the vector is built from the generated
description, not from the raw text. Search therefore matches the meaning of each
chunk. The original text is still kept and returned to you for display.

Search reverses the path. Your query is embedded and compared against the stored
vectors. You get back the closest chunks, each with its file name, position,
original text, description, and a distance score.

Indexing and search are powered by brinicle, a vector search engine written in
C++. The storage and query paths run in native code, which keeps search fast
even as a collection grows.

## Installation

```bash
pip install bicardinal
```

Audio support needs an optional dependency and the system tool `ffmpeg`:

```bash
pip install "bicardinal[audio]"
```

bicardinal requires Python 3.12 or newer.

## Provider keys

bicardinal calls external services for some steps. Set the keys for the services
you use:

```bash
export OPENAI_API_KEY=sk-...
export MISTRAL_API_KEY=...
```

You can also pass keys directly to the constructor instead of using environment
variables (see [Bicardinal](#bicardinal-1)).

Which service is used for which step:

| Step | Service | Always needed |
| --- | --- | --- |
| Plain text and Word extraction | None, runs locally | No external call |
| PDF text extraction | Mistral OCR | When ingesting PDFs |
| Image description and transcription | OpenAI vision | When ingesting images |
| Audio transcription | OpenAI transcription | When ingesting audio |
| Chunk descriptions (summaries) | OpenAI | When ingesting text, Word, PDF, or audio |
| Embeddings | sentence-transformers (local) by default, or Voyage AI | Always |

Note that ingesting any text based file (plain text, Word, PDF, audio) calls the
summarizer, so an OpenAI key is needed for those. Images do not call the
summarizer because the vision model already produces a description. The default
embedding model runs locally and needs no key; it is downloaded on first use.

## Supported file types

The file type is detected from the content of the bytes, not from the file name
extension.

| Modality | Detected types |
| --- | --- |
| Text | any `text/*` content, for example `.txt` |
| Word | `.docx` |
| PDF | `.pdf` |
| Image | PNG, JPEG, WebP, GIF |
| Audio | MP3, WAV, M4A or MP4 audio, OGG, FLAC |

A file whose type is not in this list raises `UnsupportedFileType`. A file with
zero bytes raises `EmptyFile`.

## Core concepts

**Store.** A `Bicardinal` instance owns a directory on disk and the shared models
and clients. You create one store per root directory.

**Collection.** A named group of files inside a store. Searches run within a
single collection. A store can hold many collections.

**Lifecycle.** A collection is filled in batches. The order is:

1. `init(mode)` opens the collection for adding files.
2. `ingest(...)` adds one file. Call it once per file.
3. `finalize()` builds the index so the collection can be searched.
4. `search(...)`, `search_in_file(...)`, and `most_similar_files(...)` query it.
5. `close()` releases resources.

Everything is written to disk under the store directory, so a collection can be
reopened later with `store.open(name)`.

**Scores.** A score is a distance. Smaller means closer to the query. Results are
returned in ascending order, so the first result is the best match.

## Quick start

```python
from pathlib import Path
from bicardinal import Bicardinal

store = Bicardinal("./data")     # a directory that holds collections
col = store.create("docs")       # make a new collection
col.init("build")                # open it for the first batch of files

# Add any supported file.
for path in Path("./my_files").glob("*"):
    col.ingest(path.name, path.read_bytes())

col.finalize()                   # build the index

# Search across every file in the collection.
for hit in col.search("quarterly revenue growth", k=5):
    print(f"{hit.score:.3f}  {hit.filename}#{hit.chunk_index}  {hit.raw_text[:80]}")

col.close()
```

## Use cases

### Build a searchable knowledge base from a folder of mixed files

```python
from pathlib import Path
from bicardinal import Bicardinal, ExtractionError

store = Bicardinal("./data")
col = store.create("knowledge")
col.init("build")

for path in sorted(Path("./inbox").glob("*")):
    try:
        result = col.ingest(path.name, path.read_bytes())
        print(f"{path.name}: {result.n_chunks} chunks")
    except ExtractionError as e:
        print(f"skip {path.name}: {e}")

col.finalize()
col.close()
```

### Search within a single document

Use this when you already know which file you care about and want the most
relevant passages inside it.

```python
for hit in col.search_in_file("payment terms", "contract.pdf", k=5):
    print(hit.chunk_index, hit.raw_text[:120])
```

### Read a document's chunks directly

Use this when you want a file's content in order, rather than the passages that
match a query. Useful for displaying a document, or paging through it.

```python
# All chunks of one file, in order
for c in col.read_chunks("report.pdf"):
    print(c.chunk_index, c.raw_text[:120])

# Page through a long file
first_page = col.read_chunks("report.pdf", start=0, limit=20)
next_page = col.read_chunks("report.pdf", start=20, limit=20)
```

### Rank whole files by relevance

Use this to find which documents are most about a topic, rather than which
individual passages.

```python
for f in col.most_similar_files("budget forecast", k=3):
    print(f"{f.score:.3f}  {f.filename}")
    print("   best passage:", f.best_chunk.raw_text[:100])
```

### Reopen a collection later

The collection is on disk. A new process can open it without re-adding files.

```python
store = Bicardinal("./data")
col = store.open("knowledge")
hits = col.search("renewal date")
col.close()
```

### Track progress and usage during ingestion

`ingest` accepts a callback that reports each stage, and returns a usage summary
you can use for cost accounting.

```python
def show(stage, done, total):
    print(f"  {stage:<8} {done}/{total}")

result = col.ingest("report.pdf", Path("report.pdf").read_bytes(), on_progress=show)
print("OCR pages:", result.usage.ocr_pages)
print("summary tokens in/out:",
      result.usage.summarizer_input_tokens,
      result.usage.summarizer_output_tokens)
```

### Update a document

There is no in place update. Delete the file, then add the new version.

```python
col.delete("policy.txt")
col.init("insert")
col.ingest("policy.txt", new_bytes)
col.finalize()
```

### Manage several collections

```python
store = Bicardinal("./data")
print(store.list())          # names of existing collections
store.delete("old_project")  # remove one collection
```

### Use Voyage AI embeddings instead of the local model

```python
from bicardinal import Bicardinal, Config

config = Config(
    embed_provider="voyage",
    embed_model="voyage-4",
    embed_output_dimension=1024,
)
store = Bicardinal("./data", config=config, voyage_api_key="...")
```

When `embed_provider` is `"voyage"` you must set `embed_model` to a Voyage model
and set `embed_output_dimension`. Leaving `embed_model` at the default raises a
`ValueError`.

## API reference

Public names are importable from the top level package:

```python
from bicardinal import (
    Bicardinal, Collection, Config, CollectionStatus,
    AddResult, SearchHit, FileHit, Chunk, Usage, Modality,
    BicardinalError, DuplicateDocument, DocumentNotFound,
    CollectionExists, CollectionNotFound, UnsupportedFileType,
    EmptyFile, ExtractionError,
)
```

### Bicardinal

A store that owns a root directory and the shared models. Create one per root
directory.

```python
Bicardinal(
    root_dir,
    *,
    config=None,
    openai_api_key=None,
    mistral_api_key=None,
    voyage_api_key=None,
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `root_dir` | str or Path | required | Directory that holds the collections. Created if missing. |
| `config` | Config or None | `None` | Settings for chunking, models, index, and search. A default `Config` is used when omitted. |
| `openai_api_key` | str or None | `None` | Key for OpenAI. Falls back to the `OPENAI_API_KEY` environment variable. |
| `mistral_api_key` | str or None | `None` | Key for Mistral. Falls back to the `MISTRAL_API_KEY` environment variable. |
| `voyage_api_key` | str or None | `None` | Key for Voyage AI, used only when `embed_provider` is `"voyage"`. |

Methods:

| Method | Returns | Description |
| --- | --- | --- |
| `create(name)` | Collection | Create a new collection. Raises `CollectionExists` if the name is taken. |
| `open(name)` | Collection | Open an existing collection. Raises `CollectionNotFound` if it does not exist. |
| `list()` | list of str | Names of existing collections, sorted. |
| `delete(name)` | None | Remove a collection and its files. Raises `CollectionNotFound` if it does not exist. |
| `destroy()` | None | Remove every collection in the store. |

Collection names may contain letters, digits, and underscores only. Other
characters raise `ValueError`.

### Collection

A named group of files. Obtain one from `store.create(name)` or
`store.open(name)`. A collection can be used as a context manager, which calls
`close()` on exit.

```python
with store.open("docs") as col:
    hits = col.search("topic")
```

#### init

```python
init(mode="insert")
```

Open the collection for a batch of `ingest` calls. Call this before adding
files.

| `mode` value | Use |
| --- | --- |
| `"build"` | First time you fill an empty collection. |
| `"insert"` | Add new files to a collection that already has content. |
| `"upsert"` | Add files and overwrite entries that share an identifier. |

#### ingest

```python
ingest(filename, data, *, on_progress=None) -> AddResult
```

Add one file. `data` is the raw bytes of the file. `filename` is the name you
will see in results and use to scope or delete the file later.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `filename` | str | required | Identifier for the file inside the collection. Must be unique. |
| `data` | bytes | required | Raw file content. The type is detected from these bytes. |
| `on_progress` | callable or None | `None` | Called as `on_progress(stage, done, total)` during ingestion. |

The `stage` value passed to `on_progress` is one of `"extract"`, `"describe"`,
`"embed"`, or `"write"`.

Returns an [`AddResult`](#addresult). Raises `DuplicateDocument` if the file name
already exists, `EmptyFile` for zero byte input, `UnsupportedFileType` for an
unknown type, and `ExtractionError` if extraction fails.

Per chunk description failures do not raise. They are reported in the
`errors` list of the returned `AddResult`, and the raw chunk text is used in
place of the missing description.

#### finalize

```python
finalize(*, max_retries=3) -> dict[str, str]
```

Build the index for everything added since `init`. Call this before searching.

Returns a dictionary. An empty dictionary means every file was stored. A non
empty dictionary maps a file name to the error that prevented it from being
stored after retries.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | int | `3` | How many times to retry files whose write step failed. |

#### search

```python
search(query, k=None, *, n_jobs=None, efs=None) -> list[SearchHit]
```

Search across every file in the collection.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | str | required | The text to search for. |
| `k` | int or None | `None` | Maximum number of results. Falls back to `Config.default_k` (10). |
| `n_jobs` | int or None | `None` | Parallel workers for the search. Falls back to `Config.n_jobs`. |
| `efs` | int or None | `None` | Search breadth for this query. Higher can improve recall at the cost of speed. Falls back to `Config.efs`. |

Returns a list of [`SearchHit`](#searchhit), closest first. Returns an empty list
if nothing has been finalized yet.

#### search_in_file

```python
search_in_file(query, filename, k=None, *, n_jobs=None, efs=None, exact=True) -> list[SearchHit]
```

Search only within one file.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | str | required | The text to search for. |
| `filename` | str | required | The file to search inside. |
| `k` | int or None | `None` | Maximum number of results. Falls back to `Config.default_k`. |
| `n_jobs` | int or None | `None` | Parallel workers. Falls back to `Config.n_jobs`. |
| `efs` | int or None | `None` | Search breadth. Falls back to `Config.efs`. |
| `exact` | bool | `True` | Keep only results that belong to `filename`. Leave this on unless you have a reason to change it. |

Returns a list of [`SearchHit`](#searchhit). Raises `DocumentNotFound` if the
file is not in the collection.

#### most_similar_files

```python
most_similar_files(query, k=None, *, candidate_k=100, n_jobs=None, efs=None) -> list[FileHit]
```

Rank whole files by how well their best passage matches the query.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | str | required | The text to search for. |
| `k` | int or None | `None` | Maximum number of files to return. Falls back to `Config.default_k`. |
| `candidate_k` | int | `100` | How many passages to consider before grouping them by file. Raise it if you have many files and want wider coverage. |
| `n_jobs` | int or None | `None` | Parallel workers. Falls back to `Config.n_jobs`. |
| `efs` | int or None | `None` | Search breadth. Falls back to `Config.efs`. |

Returns a list of [`FileHit`](#filehit), best file first. Returns an empty list
if nothing has been finalized yet.

#### read_chunks

```python
read_chunks(filename, *, start=0, limit=None) -> list[Chunk]
```

Read a file's stored chunks directly, in order, without a search query. Use this
when you want the document's content as it was chunked, rather than the passages
that match some query.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `filename` | str | required | The file whose chunks to read. |
| `start` | int | `0` | Index of the first chunk to return. Offsets into the ordered list. |
| `limit` | int or None | `None` | Maximum number of chunks to return. `None` returns all remaining. |

Returns a list of [`Chunk`](#chunk), ordered by `chunk_index` starting at 0.
Raises `DocumentNotFound` if the file is not in the collection. A registered file
that produced no chunks (for example a whitespace only file) returns an empty
list rather than raising.

`start` and `limit` follow normal slicing: a `start` past the end returns an
empty list, and a `limit` larger than the number of remaining chunks returns
whatever is left.

#### status

```python
status() -> CollectionStatus
```

Return counts and file names for the collection. See
[`CollectionStatus`](#collectionstatus).

#### delete

```python
delete(filename) -> None
```

Remove one file and all of its passages from the collection. Raises
`DocumentNotFound` if the file is not present.

#### close

```python
close() -> None
```

Release the collection resources. Call this when you are done, or use the
collection as a context manager.

#### destroy

```python
destroy() -> None
```

Remove all stored data for the collection.

### Config

Settings passed to `Bicardinal`. Every field has a default, so you only set what
you want to change.

```python
from bicardinal import Config
config = Config(chunk_size=512, overlap=0.1)
```

Chunking:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | int | `512` | Maximum size of each chunk, measured in tokens. |
| `overlap` | float | `0.1` | Fraction of overlap between neighboring chunks, from 0 up to but not including 1. |

Extraction models:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ocr_model` | str | `"mistral-ocr-latest"` | Model used to read PDFs. |
| `image_model` | str | `"gpt-5.4"` | Vision model used to describe and transcribe images. |
| `transcribe_model` | str | `"whisper-1"` | Model used to transcribe audio. |

Embeddings:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `embed_provider` | str | `"sentence-transformers"` | Embedding backend. Either `"sentence-transformers"` (local) or `"voyage"`. |
| `embed_model` | str | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model name. For Voyage, set this to a Voyage model. |
| `embed_output_dimension` | int or None | `None` | Vector size. Required for Voyage. The local default model determines its own size (384). |
| `embed_batch_size` | int | `64` | Number of texts embedded per batch. |
| `embed_device` | str or None | `None` | Device for the local model, for example `"cpu"` or `"cuda"`. Applies to sentence-transformers only. |
| `embed_doc_prompt` | str or None | `None` | Optional prompt prefix applied to stored text. Applies to sentence-transformers only. |
| `embed_query_prompt` | str or None | `None` | Optional prompt prefix applied to queries. Applies to sentence-transformers only. |

Summaries:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `summarizer_model` | str | `"gpt-5.4-nano"` | Model used to describe each chunk. |
| `summarizer_max_concurrency` | int | `8` | How many descriptions are requested in parallel. |
| `summarizer_reasoning_effort` | str | `"minimal"` | Reasoning effort passed to the summarizer model. |

Index:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `M` | int | `16` | Index graph connectivity. Higher uses more memory and can improve recall. |
| `ef_construction` | int | `200` | Index build breadth. Higher builds a better index more slowly. |
| `efs` | int | `64` | Default search breadth. Can be overridden per query. |
| `build_n_threads` | int | `1` | Threads used while building the index. |

Storage and search:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `shard_count` | int | `1` | Number of storage shards. |
| `default_k` | int | `10` | Default number of results when `k` is not given. |
| `file_scope_threshold` | float | `2.0` | Distance limit used when searching inside a single file. |
| `n_jobs` | int | `1` | Default parallel workers for search. Raise it when `shard_count` is above 1. |

### Result types

#### AddResult

Returned by `ingest`.

| Field | Type | Description |
| --- | --- | --- |
| `filename` | str | The file that was added. |
| `n_chunks` | int | Number of chunks created from the file. |
| `usage` | Usage | Resource use for this file. |
| `errors` | list of str | Per chunk description failures, if any. |

#### SearchHit

Returned by `search` and `search_in_file`, and held inside `FileHit`.

| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | str | Identifier of the chunk. |
| `filename` | str | File the chunk came from. |
| `chunk_index` | int | Position of the chunk within the file, starting at 0. |
| `raw_text` | str | Original chunk text. |
| `description` | str | Generated description that was embedded. |
| `score` | float | Distance from the query. Smaller is closer. |

#### FileHit

Returned by `most_similar_files`.

| Field | Type | Description |
| --- | --- | --- |
| `filename` | str | The file. |
| `score` | float | Distance of the file's best chunk. Smaller is closer. |
| `best_chunk` | SearchHit | The best matching chunk in the file. |

#### Chunk

Returned by `read_chunks`. Like a `SearchHit` without a score, since reading
chunks is not a query and has no distance.

| Field | Type | Description |
| --- | --- | --- |
| `chunk_id` | str | Identifier of the chunk. |
| `filename` | str | File the chunk came from. |
| `chunk_index` | int | Position of the chunk within the file, starting at 0. |
| `raw_text` | str | Original chunk text. |
| `description` | str | Generated description that was embedded. |


#### CollectionStatus

Returned by `status`.

| Field | Type | Description |
| --- | --- | --- |
| `n_files` | int | Number of files in the collection. |
| `n_chunks` | int | Total number of chunks across all files. |
| `filenames` | list of str | Names of the files. |

#### Usage

Resource counts for an ingestion. Two `Usage` values can be added together with
`+`.

| Field | Type | Description |
| --- | --- | --- |
| `summarizer_input_tokens` | int | Tokens sent to the summarizer. |
| `summarizer_output_tokens` | int | Tokens returned by the summarizer. |
| `image_input_tokens` | int | Tokens sent to the image model. |
| `image_output_tokens` | int | Tokens returned by the image model. |
| `audio_seconds` | float | Seconds of audio transcribed. |
| `ocr_pages` | int | Pages read by OCR. |

#### Modality

An enumeration of the file kinds: `TEXT`, `DOCX`, `PDF`, `IMAGE`, `AUDIO`.

### Exceptions

All exceptions derive from `BicardinalError`, so you can catch that one type to
handle any library error.

| Exception | Raised when |
| --- | --- |
| `BicardinalError` | Base class for the exceptions below. |
| `DuplicateDocument` | A file with this name already exists in the collection. |
| `DocumentNotFound` | No file with this name exists in the collection. |
| `UnsupportedFileType` | The detected file type has no extractor. |
| `EmptyFile` | The input was zero bytes. |
| `ExtractionError` | An extractor (OCR, transcription, decoding) failed. |
| `CollectionExists` | A collection with this name already exists. |
| `CollectionNotFound` | No collection with this name exists. |

## Scaling and performance

Indexing and search are handled by brinicle, a vector search engine written in
C++. Because the index build and the query run as native code rather than
Python, search stays fast as the number of chunks grows.

For large collections you can split the stored data across several shards by
setting `shard_count` above 1 in `Config`. Sharding spreads the stored vectors
and records, which helps when a single collection holds a large amount of data.
When you use more than one shard, raise `n_jobs` (either in `Config` or per
query) so the shards are searched in parallel.

```python
from bicardinal import Bicardinal, Config

# Spread a large collection across 8 shards and search them in parallel.
config = Config(shard_count=8, n_jobs=8)
store = Bicardinal("./data", config=config)
```

`shard_count` is fixed when the collection is created, so choose it before you
start adding files. `n_jobs` can be changed at any time, including per call to
`search`, `search_in_file`, and `most_similar_files`.

## Behavior notes

- Search matches the generated description of each chunk, while the returned
  `raw_text` is the original content. Expect results to match meaning rather
  than exact wording.
- Call `init` before adding files and `finalize` before searching. Searching a
  collection that has not been finalized returns an empty list.
- File names must be unique within a collection. Adding the same name twice
  raises `DuplicateDocument`. To replace a file, delete it and add it again.
- A file whose extracted content is empty or only whitespace is registered with
  zero chunks. It appears in `status().filenames` but produces no search
  results.
- `read_chunks` returns a file's chunks in `chunk_index` order without running a
  search. It raises `DocumentNotFound` for an unknown file, but returns an empty
  list for a registered file that has no chunks.
- Requesting more results than exist is safe. You receive at most as many
  results as there are chunks.
- Scores are distances. Sort ascending and treat the first result as the best
  match.
- The collection is stored on disk. Reopen it later with `store.open(name)`
  without adding the files again.