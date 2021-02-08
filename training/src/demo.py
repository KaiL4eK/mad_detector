import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", ".."))
sys.path.append(REPO_ROOT)

import streamlit as st

# Setup logger
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s[line:%(lineno)d] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d/%H:%M:%S",
)
logger = logging.getLogger(__name__)

from dcvt.common.utils import remote
from dcvt.detection.infer import InferDetection
from dcvt.common.utils.fs import FilepathLoader, get_images_from_directory

from dcvt.common.utils.viz import draw_detections

import cv2
import torch


PYTORCH_HASH_FUNCS = {
    torch.Tensor: id
}


@st.cache(allow_output_mutation=True)
def get_model(model_path):
    loader = FilepathLoader()
    path = loader.load_path(model_path)
    model = InferDetection.from_file(path)
    logger.debug(f"Loaded model {model.name}")
    return model


@st.cache(allow_output_mutation=True, hash_funcs=PYTORCH_HASH_FUNCS)
def get_image(image_fpaths, model, index):
    image = cv2.imread(image_fpaths[index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bboxes, label_ids, scores = model.infer_image(image)
    labels = model.map_labels(label_ids)

    draw_detections(
        image,
        bboxes,
        labels,
        scores,
        font_sz=0.4,
        line_width=1
    )

    return image


def main():
    remote.init()

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "RF20_test", "Images")
    MODEL_PATH = "s3://mlflow/artifacts/3/d50e8774086047d0a3cbf655a4768f01/artifacts/rf_signs_detection_Yolo4Tiny_best_map75.pth"
    # MODEL_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'rf_signs_detection_Yolo4Tiny_best_map75.pth')

    model = get_model(MODEL_PATH)
    image_fpaths = get_images_from_directory(DATA_PATH)

    st.title("Signs detection demo")
    st.sidebar.title("Options")
    data_index_slider = st.sidebar.slider("Data index", 0, len(image_fpaths))

    image_out = get_image(image_fpaths, model, data_index_slider)
    st.image(image_out, use_column_width=True)


if __name__ == "__main__":
    main()
