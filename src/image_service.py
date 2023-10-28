from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from neural_searcher import NeuralSearcher

app = FastAPI()
COLLECTION_NAME = "amphetamemes"

neural_searcher = NeuralSearcher(collection_name=COLLECTION_NAME)


class ImportImageRequest(BaseModel):
    path: str


class ImportImageResponse(BaseModel):
    pass  # Assuming there are no fields to be returned


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    telegram_filename: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.post("/import_image", response_model=ImportImageResponse)
async def import_image(request: ImportImageRequest):
    neural_searcher.import_image(request.path)

    return ImportImageResponse()

@app.post("/batch_inference", response_model=ImportImageResponse)
async def import_image(request: ImportImageRequest):
    neural_searcher.batch_upload(request.path, COLLECTION_NAME)

    return ImportImageResponse()


@app.get("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    search_results = neural_searcher.search(request.query)
    results = SearchResult(telegram_filename=search_results)

    return SearchResponse(results=[results])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
