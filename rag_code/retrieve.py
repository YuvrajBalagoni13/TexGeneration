import os
from .create_embeddings import CreateEmbedding
import chromadb as db

class SampleRetriever:
    def __init__(
        self,
        model_name : str = "MobileCLIP2-S0",
        pretrained : str = "dfndr2b",
        db_path : str = "chromadb",
        collection_name : str = "brick_texture_collection",
        num_results : int = 5,
        device : str = 'cpu'
    ) -> None:

        self.num_results = num_results
        self.device = device

        self.embedder = CreateEmbedding(
            model_name = model_name,
            pretrained = pretrained,
            device = device
        )

        self.client = db.PersistentClient(db_path)
        self.collection = self.client.get_collection(name = collection_name)

    def retrieve(
        self,
        image_path : str
    ):
        query_embedding = self.embedder.embed(image_path)
        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results = self.num_results
        )
        return results

if __name__ == "__main__":
    sampler = SampleRetriever()
    results = sampler.retrieve("bsdf_bricktex_dataset/test/images/ShaderNodeBsdfPrincipled_ShaderNodeTexBrick_000_092.png")

    # print(results)
    for ids in results['ids'][0]:
        print(ids)