genai-embedding-populator

> [!CAUTION]
> GCP VertexAI does not support BatchAPI, latest embeddings model is not possible to use with BathcAPI either, ComputeEmbeddings API is experimental, bad API.
> As of 2026-03-03, it is impossible to compute embeddings in Batch API.
> Until it is possible, we archiving this repo.
> https://github.com/nikolaydubina/everything-wrong-with-gcp-genai

Compute genai embeddings and store them in Firestore.

- [x] Batch mode
- [x] Redis (pending documents set)
- [x] Firestore (source, destination)
