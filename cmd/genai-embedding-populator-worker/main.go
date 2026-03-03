package main

import (
	"context"
	"encoding/json/v2"
	"errors"
	"flag"
	"log"
	"log/slog"
	"os"
	"strings"

	"cloud.google.com/go/firestore"
	"cloud.google.com/go/pubsub/v2"
	"cloud.google.com/go/storage"
	"github.com/ndx-technologies/genai-embedding-populator/genaix"
)

func main() {
	var (
		projectID              string
		pubsubTopic            string
		firestoreCollection    string
		firestoreEmbeddingPath string
		cloudStorageBucket     string
	)
	flag.StringVar(&projectID, "project_id", os.Getenv("PROJECT_ID"), "GCP project ID")
	flag.StringVar(&pubsubTopic, "pubsub_sub", "", "Pub/Sub subscription for Cloud Storage notifications of predictions")
	flag.StringVar(&firestoreCollection, "firestore_collection", "", "Firestore collection to write embeddings into")
	flag.StringVar(&firestoreEmbeddingPath, "firestore_embedding_path", "", "Firestore field to write embedding values")
	flag.StringVar(&cloudStorageBucket, "cloud_storage_bucket", "", "GCS bucket containing prediction results")
	flag.Parse()

	if projectID == "" || pubsubTopic == "" || firestoreCollection == "" || firestoreEmbeddingPath == "" || cloudStorageBucket == "" {
		flag.Usage()
		log.Fatal("missing required args")
	}

	ctx := context.Background()

	firestoreClient, err := firestore.NewClient(ctx, projectID)
	if err != nil {
		log.Fatal(err)
	}
	defer firestoreClient.Close()

	gcsClient, err := storage.NewClient(ctx)
	if err != nil {
		log.Fatal(err)
	}

	pubsubClient, err := pubsub.NewClient(ctx, projectID)
	if err != nil {
		log.Fatal(err)
	}
	defer pubsubClient.Close()

	resultIterator := genaix.EmbeddingBatchResultIterator{Bucket: gcsClient.Bucket(cloudStorageBucket)}

	slog.InfoContext(ctx, "starting worker", "topic", pubsubTopic)

	if err := pubsubClient.Subscriber(pubsubTopic).Receive(ctx, func(ctx context.Context, m *pubsub.Message) {
		var e struct {
			Bucket string `json:"bucket"`
			Name   string `json:"name"`
		}
		if err := json.Unmarshal(m.Data, &e); err != nil {
			slog.ErrorContext(ctx, "cannot decode message", "error", err)
			m.Nack()
			return
		}

		if !strings.HasSuffix(e.Name, "predictions.jsonl") {
			m.Ack()
			return
		}

		for docID, embedding := range resultIterator.Iter(ctx, e.Name) {
			if _, err := firestoreClient.Collection(firestoreCollection).Doc(string(docID)).Set(
				ctx,
				map[string]any{firestoreEmbeddingPath: firestore.Vector32(embedding)},
				firestore.MergeAll,
			); err != nil {
				slog.ErrorContext(ctx, "failed to write embedding", "error", err, "doc_id", docID)
			}
		}

		m.Ack()
	}); err != nil && !errors.Is(err, context.Canceled) {
		log.Fatal(err)
	}

	slog.InfoContext(ctx, "worker stopped: ok")
}
