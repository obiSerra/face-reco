from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import argparse


def get_embeddings(img_name):
    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained="vggface2").eval()

    img = Image.open(img_name)

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)

    img_embedding = resnet(img_cropped.unsqueeze(0))
    emb_out = str(img_embedding[0].detach().numpy().reshape(-1))
    emb_out = emb_out.replace("[", "")
    emb_out = emb_out.replace("]", "")

    print(emb_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save embeddings")
    parser.add_argument("input_image", help="the input image")

    args = parser.parse_args()
    get_embeddings(args.input_image)
