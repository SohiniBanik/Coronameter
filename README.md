gcloud builds submit --tag gcr.io/covid-detector-329911/COVIDAPP --project=covid-detector-329911

gcloud run deploy --image gcr.io/covid-detector-329911/COVIDAPP --platform managed --project=covid-detector-329911 --allow-unauthenticated