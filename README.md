gcloud builds submit --tag gcr.io/dse6000-project/DSE6000-Project  --project=dse6000-project

gcloud run deploy --image gcr.io/dse6000-project/DSE6000-Project --platform managed  --project=dse6000-project --allow-unauthenticated