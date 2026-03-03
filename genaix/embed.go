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

type EmbeddingName string

type EmbeddingTaskType string

const (
	Classification     EmbeddingTaskType = "CLASSIFICATION"
	Clustering         EmbeddingTaskType = "CLUSTERING"
	RetrievalDocument  EmbeddingTaskType = "RETRIEVAL_DOCUMENT"
	RetrievalQuery     EmbeddingTaskType = "RETRIEVAL_QUERY"
	QuestionAnswering  EmbeddingTaskType = "QUESTION_ANSWERING"
	FactVerification   EmbeddingTaskType = "FACT_VERIFICATION"
	CodeRetrievalQuery EmbeddingTaskType = "CODE_RETRIEVAL_QUERY"
	SemanticSimilarity EmbeddingTaskType = "SEMANTIC_SIMILARITY"
)

// https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/batch-prediction-genai-embeddings
type EmbeddingBatchRequest struct {
	Content  *genai.Content    `json:"content"`
	TaskType EmbeddingTaskType `json:"taskType,omitempty"`
}

// https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/batch-prediction-genai-embeddings
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

func (s *EmbeddingBatchSubmitter) Write(ctx context.Context, id DocumentID, text string, name EmbeddingName, task EmbeddingTaskType) error {
	if s.W == nil {
		s.W = s.Bucket.Object(s.keyRequest()).NewWriter(ctx)
	}

	req := struct {
		ID      RequestID             `json:"id"`
		DocID   DocumentID            `json:"doc_id"`
		Name    EmbeddingName         `json:"embedding_name,omitempty"`
		Request EmbeddingBatchRequest `json:"request"`
	}{
		ID:    NewRequestID(),
		DocID: id,
		Name:  name,
		Request: EmbeddingBatchRequest{
			Content:  &genai.Content{Role: "user", Parts: []*genai.Part{genai.NewPartFromText(text)}},
			TaskType: task,
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
	_, err := s.Client.Batches.Create(
		ctx,
		s.Model,
		&genai.BatchJobSource{Format: "jsonl", GCSURI: []string{"gs://" + s.Bucket.BucketName() + "/" + s.keyRequest()}},
		&genai.CreateBatchJobConfig{
			DisplayName: "embedding_populator_" + s.ID.String(),
			Dest:        &genai.BatchJobDestination{Format: "jsonl", GCSURI: "gs://" + s.Bucket.BucketName() + "/" + s.keyResult()},
		},
	)
	return err
}

type EmbeddingResult struct {
	ID            RequestID              `json:"id"`
	DocID         DocumentID             `json:"doc_id"`
	EmbeddingName EmbeddingName          `json:"embedding_name,omitempty"`
	Response      EmbeddingBatchResponse `json:"response"`
}

type EmbeddingBatchResultIterator struct{ Bucket *storage.BucketHandle }

func (s EmbeddingBatchResultIterator) Iter(ctx context.Context, key string) iter.Seq[EmbeddingResult] {
	return func(yield func(EmbeddingResult) bool) {
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
			var e EmbeddingResult
			if err := json.Unmarshal(scanner.Bytes(), &e); err != nil {
				slog.ErrorContext(ctx, "failed to unmarshal or empty embedding response", "error", err, "id", e.ID, "content", string(scanner.Bytes()))
				continue
			}
			if !yield(e) {
				return
			}
		}
	}
}
