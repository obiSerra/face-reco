from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch


img_name = "matteo.jpg"

IMG_DIR = "images"
EMB_DIR = "embedding"


if __name__ == "__main__":
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained="vggface2").eval()

    img = Image.open(f"{IMG_DIR}/{img_name}")

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=f"{IMG_DIR}/cropped_{img_name}")
    # img_cropped = mtcnn(img)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    emb_out = str(img_embedding[0].detach().numpy().reshape(-1))
    emb_out = emb_out.replace("[", "")
    emb_out = emb_out.replace("]", "")

    embedding_name = img_name.replace(".jpg", "")

    # with open(f"{EMB_DIR}/embedding_{embedding_name}.txt", "w") as f:
    #     f.write(emb_out)

    print(emb_out)

    # Or, if using for VGGFace2 classification
    # resnet.classify = True
    # img_probs = resnet(img_cropped.unsqueeze(0))
