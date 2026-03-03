package genaix

import (
	"bufio"
	"context"
	"encoding/json/v2"
	"io"
	"iter"
	"log/slog"

	"cloud.google.com/go/storage"
	"github.com/google/uuid"
	"google.golang.org/genai"
)

type DocumentID string

type RequestID struct{ uuid.UUID }

func NewRequestID() RequestID { return RequestID{uuid.New()} }

type EmbeddingBatchRequest struct {
	Model   string         `json:"model"`
	Content *genai.Content `json:"content"`
}

type EmbeddingBatchResponse struct {
	Embeddings []EmbeddingValues `json:"embeddings"`
}

type EmbeddingValues struct {
	Values []float32 `json:"values"`
}

type EmbeddingBatchSubmitter struct {
	ID          RequestID
	Model       string
	Client      *genai.Client
	Bucket      *storage.BucketHandle
	RequestPath string
	Count       int
	W           io.WriteCloser
}

func (s *EmbeddingBatchSubmitter) keyRequest() string {
	return s.RequestPath + "/" + s.ID.String() + ".jsonl"
}

func (s *EmbeddingBatchSubmitter) keyResult() string {
	return s.RequestPath + "/result/" + s.ID.String()
}

func (s *EmbeddingBatchSubmitter) Write(ctx context.Context, id DocumentID, text string) error {
	if s.W == nil {
		s.W = s.Bucket.Object(s.keyRequest()).NewWriter(ctx)
	}

	req := struct {
		ID      string                `json:"id"`
		DocID   string                `json:"doc_id"`
		Request EmbeddingBatchRequest `json:"request"`
	}{
		ID:    NewRequestID().String(),
		DocID: string(id),
		Request: EmbeddingBatchRequest{
			Model:   s.Model,
			Content: &genai.Content{Role: "user", Parts: []*genai.Part{genai.NewPartFromText(text)}},
		},
	}

	if err := json.MarshalWrite(s.W, req); err != nil {
		return err
	}

	s.W.Write([]byte("\n"))
	s.Count++
	return nil
}

func (s *EmbeddingBatchSubmitter) Submit(ctx context.Context) error {
	if s.W == nil {
		return nil
	}
	if err := s.W.Close(); err != nil {
		return err
	}
	if s.Count == 0 {
		return nil
	}
	src := genai.BatchJobSource{
		Format: "jsonl",
		GCSURI: []string{"gs://" + s.Bucket.BucketName() + "/" + s.keyRequest()},
	}
	dst := genai.BatchJobDestination{
		Format: "jsonl",
		GCSURI: "gs://" + s.Bucket.BucketName() + "/" + s.keyResult(),
	}
	_, err := s.Client.Batches.Create(ctx, s.Model, &src, &genai.CreateBatchJobConfig{
		DisplayName: "embedding_populator_" + s.ID.String(),
		Dest:        &dst,
	})
	return err
}

type EmbeddingBatchResultIterator struct {
	Bucket *storage.BucketHandle
}

func (s EmbeddingBatchResultIterator) Iter(ctx context.Context, key string) iter.Seq2[DocumentID, []float32] {
	return func(yield func(DocumentID, []float32) bool) {
		r, err := s.Bucket.Object(key).NewReader(ctx)
		if err != nil {
			slog.ErrorContext(ctx, "failed to open embedding results", "error", err, "key", key)
			return
		}
		defer r.Close()

		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 0, 5*1024*1024), 16*1024*1024)

		defer func() {
			if err := scanner.Err(); err != nil {
				slog.ErrorContext(ctx, "error reading embedding results", "error", err, "key", key)
			}
		}()

		for scanner.Scan() {
			var e struct {
				ID       RequestID               `json:"id"`
				DocID    DocumentID              `json:"doc_id"`
				Response *EmbeddingBatchResponse `json:"response"`
			}

			if err := json.Unmarshal(scanner.Bytes(), &e); err != nil || e.Response == nil || len(e.Response.Embeddings) == 0 {
				slog.ErrorContext(ctx, "failed to unmarshal or empty embedding response", "error", err, "id", e.ID)
				continue
			}

			if !yield(e.DocID, e.Response.Embeddings[0].Values) {
				return
			}
		}
	}
}
