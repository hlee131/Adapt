from env.scene import example_scene 
from src.utils import DEVICE

from sentence_transformers import SentenceTransformer
import torch

if __name__ == '__main__':
    failed = set()
    embedding_dict = dict()
    sentence_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device=DEVICE)

    for _, v in example_scene.items():
        desc = v["description"]
        print(f"Embedding {desc}")
        if desc not in embedding_dict and desc not in failed:
            try:
                embedding_dict[v] = sentence_model.encode(v, batch_size=1, convert_to_tensor=True).detach()
            except:
                print(f"Failed to embed {desc}")
                failed.add(desc)

    print(len(embedding_dict))
    print(len(failed))
    print(len(example_scene))
    torch.save(embedding_dict, "models/_sentence_embeddings_minilm.pt")
