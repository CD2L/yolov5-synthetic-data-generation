from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import string
import shutil
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


dim = 1000
n_samples = 5
n_texts = 12

white = (255, 255, 255)
black = (0, 0, 0)

if os.path.exists("./data/"):
    shutil.rmtree("./data/")
os.mkdir("./data/")
os.mkdir("./data/images")
os.mkdir("./data/labels")

with open(f"data/labelmap.pbtxt", "a") as f:
    f.write(
        """item {
        name: "text-block",
        id: 0,
        display_name: "text-block"
}
    """
    )


def make_text(
    n_words: int = 20,
    n_words_per_line: int = 8,
    word_size_low: int = 3,
    word_size_high: int = 5,
) -> str:
    text = " ".join(
        "".join(
            random.choices(
                string.ascii_uppercase + string.ascii_lowercase + string.digits,
                k=random.randint(word_size_low, word_size_high),
            )
        )
        for _ in range(n_words)
    ).split(" ")

    for i, _ in enumerate(text):
        if not i % n_words_per_line and i:
            text.insert(i - 1, "\n")
    return " ".join(text)


def iou(boxA, boxB):
    """taken from: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)


for i in tqdm(range(n_samples), ascii=True):
    image = Image.new(mode="RGB", size=(dim, dim), color=white)
    boxes = []

    for j in range(n_texts):
        mask = Image.new(mode="RGBA", size=(dim, dim), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(mask)

        xy = random.choices(np.linspace(0, mask.height, 100).astype(np.int64), k=2)

        draw = ImageDraw.Draw(mask)
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text(xy, text := make_text(25, 5), black, font=font, anchor="mm")

        draw.rectangle(mask.getbbox(), outline="red", fill=None)
        x0, y0, x1, y1 = mask.getbbox()

        width, height = abs(x0 - x1), abs(y0 - y1)
        x_center = (x0 + 0.5 * (width)) / mask.width
        y_center = (y0 + 0.5 * (height)) / mask.height
        width /= mask.width
        height /= mask.height

        boxes.append(mask.getbbox())
        cond = np.asarray([iou(boxes[-1], b) for i, b in enumerate(boxes)])

        if not j or (cond > 0.13).sum() <= 1:
            with open(f"data/labels/{i+1}.txt", "a") as f:
                f.write("{} {} {} {} {}\n".format(0, x_center, y_center, width, height))
            image.paste(mask, None, mask)
            j += 1

    image.save(f"data/images/{i+1}.png", format="PNG", quality=95, subsampling=0)
