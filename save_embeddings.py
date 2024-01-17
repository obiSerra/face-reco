from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse


img_name = "robi.jpg"

IMG_DIR = "images"
EMB_DIR = "embedding"


def gen_embeddings(img_name, embedding_file, cropped_image=None):
    mtcnn = MTCNN(image_size=160)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained="vggface2").eval()

    img = Image.open(img_name)

    img_cropped = mtcnn(img, save_path=cropped_image)

    img_embedding = resnet(img_cropped.unsqueeze(0))
    emb_out = str(img_embedding[0].detach().numpy().reshape(-1))
    emb_out = emb_out.replace("[", "")
    emb_out = emb_out.replace("]", "")

    with open(embedding_file, "w") as f:
        f.write(emb_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save embeddings")
    parser.add_argument("input_image", help="the input image")

    parser.add_argument("-o", "--output-file", metavar="output_embedding", help="the output embedding file", required=True)
    parser.add_argument("-c", "--cropped-image", metavar="cropped_image", help="the cropped image")

    args = parser.parse_args()
    gen_embeddings(args.input_image, args.output_file, args.cropped_image)
