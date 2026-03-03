package main

import (
	"context"
	"flag"
	"log"
	"log/slog"
	"os"
	"strings"

	"cloud.google.com/go/firestore"
	"cloud.google.com/go/storage"
	"github.com/ndx-technologies/genai-embedding-populator/genaix"
	"github.com/redis/go-redis/v9"
	"google.golang.org/api/iterator"
	"google.golang.org/genai"
	grpccodes "google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"
)

func main() {
	var (
		projectID               string
		firestoreCollection     string
		firestoreKeyPath        string
		redisURL                string
		redisSetKey             string
		cloudStorageBucket      string
		cloudStorageRequestPath string
		genaiLocation           string
		genaiModel              string
	)
	flag.StringVar(&projectID, "project_id", os.Getenv("PROJECT_ID"), "GCP project ID")
	flag.StringVar(&firestoreCollection, "firestore_collection", "", "Firestore collection")
	flag.StringVar(&firestoreKeyPath, "firestore_key_path", "", "Firestore field path (dot-separated) to compute embedding from")
	flag.StringVar(&redisURL, "redis", "", "Redis URL (required when redis_set_key is set)")
	flag.StringVar(&redisSetKey, "redis_set_key", "", "Redis set key of doc IDs to process; if absent all docs in collection are processed")
	flag.StringVar(&cloudStorageBucket, "cloud_storage_bucket", "", "GCS bucket for batch requests")
	flag.StringVar(&cloudStorageRequestPath, "cloud_storage_request_path", "", "GCS path prefix for batch request files")
	flag.StringVar(&genaiLocation, "genai_location", "", "GenAI location")
	flag.StringVar(&genaiModel, "genai_model", "", "GenAI embedding model")
	flag.Parse()

	if projectID == "" || firestoreCollection == "" || firestoreKeyPath == "" || cloudStorageBucket == "" || cloudStorageRequestPath == "" {
		flag.Usage()
		log.Fatal("missing required args")
	}
	if redisSetKey != "" && redisURL == "" {
		flag.Usage()
		log.Fatal("-redis is required when -redis_set_key is set")
	}

	firestoreFieldPath := firestore.FieldPath(strings.Split(firestoreKeyPath, "."))

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

	genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
		Project:  projectID,
		Location: genaiLocation,
		Backend:  genai.BackendVertexAI,
	})
	if err != nil {
		log.Fatal(err)
	}

	submitter := &genaix.EmbeddingBatchSubmitter{
		ID:          genaix.NewRequestID(),
		Model:       genaiModel,
		Client:      genaiClient,
		Bucket:      gcsClient.Bucket(cloudStorageBucket),
		RequestPath: cloudStorageRequestPath,
	}

	if redisSetKey != "" {
		redisConfig, err := redis.ParseURL(redisURL)
		if err != nil {
			log.Fatal(err)
		}
		rdb := redis.NewClient(redisConfig)

		keys, err := rdb.SMembers(ctx, redisSetKey).Result()
		if err != nil {
			log.Fatal(err)
		}

		for _, key := range keys {
			doc, err := firestoreClient.Collection(firestoreCollection).Doc(key).Get(ctx)
			if err != nil {
				if grpcstatus.Code(err) == grpccodes.NotFound {
					continue
				}
				log.Fatal(err)
			}

			v, err := doc.DataAtPath(firestoreFieldPath)
			if err != nil {
				continue
			}
			text, _ := v.(string)
			if text == "" {
				continue
			}

			if err := submitter.Write(ctx, genaix.DocumentID(doc.Ref.ID), text); err != nil {
				log.Fatal(err)
			}
		}
	} else {
		it := firestoreClient.Collection(firestoreCollection).Documents(ctx)
		defer it.Stop()

		for {
			doc, err := it.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			v, err := doc.DataAtPath(firestoreFieldPath)
			if err != nil {
				continue
			}

			text, _ := v.(string)
			if text == "" {
				continue
			}

			if err := submitter.Write(ctx, genaix.DocumentID(doc.Ref.ID), text); err != nil {
				log.Fatal(err)
			}
		}
	}

	if err := submitter.Submit(ctx); err != nil {
		log.Fatal(err)
	}

	slog.InfoContext(ctx, "ok, batch job submitted")
}
