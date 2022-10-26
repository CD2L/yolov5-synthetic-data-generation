import os
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

PATH_PROCESSED_IMAGES = Path("../../data/processed/images/")
PATH_PROCESSED_LABELS = Path("../../data/processed/labels/")
PATH_DATA_RAW = Path("../../data/raw/")
PATH_TAGS = Path(f"{PATH_DATA_RAW}/tags/")
PATH_IMAGES = Path(f"{PATH_DATA_RAW}/images/")

NUM_RANDOM_SAMPLES = 2

try:
    os.mkdir(PATH_TAGS)
except OSError:
    print(f"ERROR: File already exists: `{PATH_TAGS}`.")

# for idx, img_path in enumerate(PATH_TAGS.iterdir()):
#     os.rename(img_path, r"{}/tag_{}.png".format(PATH_TAGS, idx+1))
for idx, img_path in enumerate(PATH_IMAGES.iterdir()):
    os.rename(img_path, r"{}/image_{}.png".format(PATH_IMAGES, idx))

tags_path = [tp for tp in PATH_TAGS.iterdir()]

counter = 1

from sys import exit

for idx, img_path in enumerate(PATH_IMAGES.iterdir()):
    # print(f"INFO: Processing image: `{img_path}`.")
    img = Image.open(img_path).convert("RGBA").resize((640, 640))

    for sp in tqdm(range(NUM_RANDOM_SAMPLES), ascii=True, desc=f"{img_path} "):
        OUTPUT_NAME = f"{counter}"
        img_cp = img.copy()
        random_tags = default_rng().choice(
            [i for i in range(1, len(tags_path) + 1)],
            size=np.random.randint(1, 4, size=(1,)),
            replace=False,
        )

        # tags_to_imgs = [Image.open(tag).convert("RGBA") for tag in tags]
        for i, class_number in enumerate(random_tags):
            folder_path = Path(f"{PATH_TAGS}/{class_number}")
            tag_img_path = [tp for tp in folder_path.iterdir()]
            random_tag_img = default_rng().choice(len(tag_img_path), replace=False)

            tag = Image.open(list(folder_path.iterdir())[random_tag_img - 1]).convert(
                "RGBA"
            )

            size = np.random.randint(low=50, high=100)
            tag = tag.resize((size, size)).rotate(
                np.random.randint(-90, 90), expand=True
            )

            x = np.random.randint(
                low=int(0.25 * img.width), high=int((1 - 0.25) * img.width)
            )
            y = np.random.randint(
                low=int(0.25 * img.height), high=int((1 - 0.25) * img.height)
            )

            x_norm = (x + 0.5 * tag.width) / img.width
            y_norm = (y + 0.5 * tag.height) / img.height
            width_norm = size / img.width
            height_norm = size / img.height

            with open(f"{PATH_PROCESSED_LABELS}/image_{OUTPUT_NAME}.txt", "a") as f:
                f.write(
                    "{} {} {} {} {}\n".format(
                        class_number, x_norm, y_norm, width_norm, height_norm
                    )
                )
            img_cp.alpha_composite(tag, (x, y))

        img_cp.save(f"{PATH_PROCESSED_IMAGES}/image_{OUTPUT_NAME}.png")
        counter += 1
