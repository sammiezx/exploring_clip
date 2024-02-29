import fiftyone
from PIL import Image
import torch
import clip

download_dir = "/home/samir/Desktop/medium/exploring_clip/downloads"
dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v7",
              split="validation",
              label_types=["detections"],
              classes=["Cat", "Dog"],
              max_samples=100,
              dataset_dir=download_dir
          )

sample = dataset.first()
image_path = sample.filepath
image = Image.open(image_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

#NOTE "image" variable is extracted from the dataset above
image = preprocess(image).unsqueeze(0).to(device)

text = "a dog"
text = ["This is a picture of " + text, "This is not a picture of " + text]
text = clip.tokenize(text).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("PROBABLITIES: " + str(probs))
    print("MATCHES: " + str((probs[0][0] > probs[0][1])))