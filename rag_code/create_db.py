import torch
from .create_embeddings import CreateEmbedding
import chromadb as db 
import os
from tqdm.auto import tqdm

DB_PATH = "chromadb"
os.makedirs(DB_PATH, exist_ok=True)

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    else:
        print(f"found {path}")

def get_data_lists(dataset_path: str):
    check_path(dataset_path)
    image_dataset_path = os.path.join(dataset_path, "images")
    json_dataset_path = os.path.join(dataset_path, "json")

    check_path(image_dataset_path)
    check_path(json_dataset_path)

    embedder = CreateEmbedding()
    img_list = os.listdir(image_dataset_path)

    ids, embeddings, metadatas, documents = [], [], [], []
    for img in tqdm(img_list):
        base_name = img.split(".")[0]
        img_path = os.path.join(image_dataset_path, img)
        json_path = os.path.join(json_dataset_path, f"{base_name}.json")
        
        if not os.path.exists(json_path) or not os.path.exists(img_path):
            print(f"skipping {base_name}...")
            continue

        with open(json_path, "r") as f:
            json_string = f.read()

        img_embedding = embedder.embed(img_path)

        ids.append(base_name)
        embeddings.append(img_embedding)
        documents.append(json_string)
        metadatas.append({
            "material_name" : base_name,
            "image_path" : img_path
        })
    
    return ids, embeddings, documents, metadatas


def main():
    print("Creating ChromaDB Vector Database...")
    client = db.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name = "brick_texture_collection",
        embedding_function = None,
        metadata = {
            "description" : "collection of brick texture images & node graphs",
            "hnsw:space" : "cosine"
        }
    )

    ids, embeddings, documents, metadatas = get_data_lists(dataset_path = "bsdf_bricktex_dataset/db")

    collection.upsert(
        ids = ids,
        embeddings = embeddings,
        metadatas = metadatas,
        documents = documents,
    )

    print(f"\nSuccessfully created & stored material dataset in {DB_PATH}")


if __name__ == "__main__":
    main()