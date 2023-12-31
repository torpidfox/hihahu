import os
import uuid
from typing import List

from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer

QDRANT_HOST_VAR = "QDRANT_HOST"
QDRANT_PORT_VAR = "QDRANT_PORT"
EMBEDDING_SIZE = 512

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer('clip-ViT-B-32', device='cpu')
        # initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=os.environ.get(QDRANT_HOST_VAR),
            port=os.environ.get(QDRANT_PORT_VAR)
        )

    def import_image(self, path: str):
        image = Image.open(path)
        img_emb = self.model.encode(image)

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                vectors=[img_emb.tolist()],
                payloads=[{"path": path}],
                ids=[str(uuid.uuid4())]
            )
        )

    def search(self, query: str) -> List[str]:
        query_emb = self.model.encode(query)
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_emb.tolist(),
            limit=5
        )

        print(search_result)

        return [el.payload['path'] for el in search_result]

    # def create_collection(self):
    #     self.qdrant_client.recreate_collection(
    #         collection_name="{collection_name}",
    #         vectors_config=models.VectorParams(size=EMBEDDING_SIZE, distance=models.Distance.COSINE),
    #         optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000)
    #     )

    def batch_upload(self, path: str, collection_name: str):
        paths = [os.path.join(path, p) for p in os.listdir(path)]
        images = [Image.open(p) for p in paths]

        img_embs = self.model.encode(images)

        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                vectors=img_embs.tolist(),
                payloads=[{"path": p} for p in paths],
                ids=[str(uuid.uuid4()) for _ in paths]
            )
        )


