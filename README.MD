## Run with Docker

### Requirements
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Steps

# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/anomalib-cfa-api.git
cd anomalib-cfa-api

# 2. Gen a dataset like this format
data/
├── train/OKImages/
└── test/
    ├── OKImages/
    └── NGImages/

# 3. Build and run
docker compose up --build

# 4. Open Swagger UI
http://localhost:8000/docs

# 5. Set dataset path in Swagger UI
dataset_path = /data