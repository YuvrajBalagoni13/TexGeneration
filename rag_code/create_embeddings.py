import torch
from PIL import Image
import open_clip
import torch.nn.functional as F

class CreateEmbedding:
    def __init__(
        self,
        model_name : str = "MobileCLIP2-S0",
        pretrained : str = "dfndr2b",
        device : str = 'cpu'
        ) -> None:

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Loading {model_name} on {self.device}...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name = model_name,
            pretrained = pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()     

    def embed(
        self,
        image_path: str,
        as_list: bool = True
    ):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        if as_list:
            return image_features.cpu().numpy().flatten().tolist()
        return image_features
        