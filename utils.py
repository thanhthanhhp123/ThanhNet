import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm
from PIL import Image
import torchvision
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)

    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = np.load(image_path)
        image = Image.fromarray(image)
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = np.load(mask_path).astype(np.uint8)
                mask = Image.fromarray(mask)
                mask = mask_transform(mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        # np.save(savename + "_segmentation.npy", segmentation)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        sns.heatmap(np.squeeze(image.transpose(1, 2, 0)), ax=axes[0], cbar=False, xticklabels=False, yticklabels=False,
                    vmin = -2, vmax = 2, cmap = 'rainbow')
        axes[0].set_title("Image")
        sns.heatmap(np.squeeze(mask.transpose(1, 2, 0)), ax=axes[1], cbar=False, xticklabels=False, yticklabels=False)
        axes[1].set_title("Mask")
        sns.heatmap(segmentation, ax=axes[2], cbar=False, xticklabels=False, yticklabels=False, cmap = 'hot')
        axes[2].set_title("Segmentation")
        # axes[0].imshow(image.transpose(1, 2, 0), vmin = -2, vmax = 2, cmap = 'rainbow')
        # axes[1].imshow(mask.transpose(1, 2, 0))
        # axes[2].imshow(segmentation, cmap = 'hot')
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        f.savefig(savename + ".png")
        plt.close()


def create_storage_folder(
    main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder, run_name)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics

def save_images(images, images_path, data):
    images = Image.fromarray(images)
    label = data["label"][0]
    file_name = data['path'][0].split('/')[-1]
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    image_paths = os.path.join(images_path, label + file_name)
    images.save(image_paths)

def convert2img(image, imgtype = np.uint8):
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.data
        else:
            return image
        image = image.cpu().numpy()
    if image.dtype != imgtype:
        image = (np.transpose(image.squeeze(), (1, 2, 0)) * 0.5 + 0.5) * 255
    return image.astype(imgtype)

def plt_show(img):
    img = torchvision.utils.make_grid(img.cpu().numpy())
    img = img.numpy()
    if img.dtype != "uint8":
        img_numpy = img * 0.5 + 0.5
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()

def compare_images(real_img, generated_img, threshold):
    generated_img = generated_img.type_as(real_img)
    diff_img = np.abs(real_img - generated_img)
    real_img = convert2img(real_img)
    generated_img = convert2img(generated_img)
    diff_img = convert2img(diff_img)

    threshold = (threshold * 0.5 + 0.5)*255
    diff_img[diff_img < threshold] = 0
    anomaly_img = np.zeros_like(real_img)
    anomaly_img[:, :, :] = real_img
    anomaly_img[np.where(diff_img > 0)[0], np.where(diff_img > 0)[1]] = [200, 0, 0]

    fig, ax = plt.subplots(1, 4, figsize=(20, 20))
    fig.set_figwidth(9)
    fig.set_tight_layout(True)
    ax = ax.reshape(-1)
    ax[0].imshow(real_img, label='real')
    ax[1].imshow(generated_img, label='generated')
    ax[2].imshow(diff_img, label='diff')
    ax[3].imshow(anomaly_img, label='anomaly')

    ax[0].set_title('real')
    ax[1].set_title('generated')
    ax[2].set_title('diff')
    ax[3].set_title('anomaly')
    plt.show()
    return convert2img(anomaly_img)