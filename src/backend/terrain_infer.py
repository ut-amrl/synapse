from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from torch import nn
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import rosbag
from PIL import Image
from config.ALL_TERRAINS_MAPS import DATASETINTstr_TO_DATASETLABELS
import cv2
from cv_bridge import CvBridge
from simple_colors import green
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True


def remove_username(dataset_name):
    return dataset_name.split("/")[-1]


class TerrainSegFormer:
    DISTINCT_COLORS_51 = [  # rgb
        (0, 0, 0),
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64),
        (64, 64, 0), (64, 0, 64), (0, 64, 64),
        (192, 0, 0), (0, 192, 0), (0, 0, 192),
        (192, 192, 0), (192, 0, 192), (0, 192, 192),
        (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255),
        (64, 192, 0), (192, 64, 0), (0, 64, 192),
        (0, 192, 64), (192, 0, 64), (64, 0, 192),
        (128, 64, 0), (128, 0, 64), (64, 128, 0),
        (0, 128, 64), (64, 0, 128), (0, 64, 128),
        (255, 192, 0), (255, 0, 192), (192, 255, 0),
        (0, 255, 192), (192, 0, 255), (0, 192, 255),
        (64, 64, 64), (192, 192, 192)
    ]

    def __init__(self,
                 cuda: str = "0",
                 hf_dataset_name: str = "sam1120/safety-utcustom-terrain-jackal-full-391",
                 hf_model_name: str = "sam1120/safety-utcustom-terrain",  # provide either this,
                 hf_model_ver: int = 0,  # or this. Other one should be made None
                 hf_username: str = "sam1120",
                 segformer_model_id: int = 5,
                 train_percentd: float = 0.70,
                 val_percentd: float = 0.30):
        self.hf_key = os.environ.get("HF_API_KEY")
        self.cuda = cuda
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_dataset_name = remove_username(hf_dataset_name)
        self.hf_model_name = remove_username(hf_model_name)
        self.hf_username = hf_username
        self.segformer_model_id = segformer_model_id
        self.hf_model_ver = hf_model_ver
        self.train_percentd = train_percentd  # DO NOT CHANGE THIS TO HAVE SAME test_ds
        self.val_percentd = val_percentd  # DO NOT CHANGE THIS TO HAVE SAME test_ds

        self.hf_dataset_identifier = f"{self.hf_username}/{self.hf_dataset_name}"
        self.hub_model_id = self.hf_model_name
        if self.hub_model_id is None:
            self.hub_model_id = f"segformer-b{self.segformer_model_id}-finetuned-{self.hf_dataset_name}-v{self.hf_model_ver}"

        self.id2label = {int(k): v for k, v in DATASETINTstr_TO_DATASETLABELS.items()}
        self.num_labels = len(self.id2label)

        self.metric = evaluate.load("mean_iou")

    def load_model_inference(self):
        try:
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(f"{self.hf_username}/{self.hub_model_id}", use_safetensors=True, use_auth_token=self.hf_key)
            self.model = SegformerForSemanticSegmentation.from_pretrained(f"{self.hf_username}/{self.hub_model_id}", use_safetensors=True, use_auth_token=self.hf_key)
        except:
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(f"{self.hf_username}/{self.hub_model_id}", use_auth_token=self.hf_key)
            self.model = SegformerForSemanticSegmentation.from_pretrained(f"{self.hf_username}/{self.hub_model_id}", use_auth_token=self.hf_key)
        self.model.eval()

    def _internal_predict(self, images, image_size):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )

        # Second, apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)
        return pred_seg

    def predict_one(self, pred_ds, idx):
        # assert 0 <= idx < len(pred_ds), f"idx must be in range [0, {len(pred_ds)})"
        image = pred_ds[idx]['pixel_values']
        gt_seg = pred_ds[idx]['labels']
        pred_seg = self._internal_predict(image, image.size[::-1])[0]
        pred_img = TerrainSegFormer.get_seg_overlay(image, pred_seg)
        gt_img = TerrainSegFormer.get_seg_overlay(image, np.array(gt_seg))
        return pred_img, gt_img, pred_seg, gt_seg

    def predict_new_with_depth(self, image, depth_image):
        image_size = image.size[::-1]
        inputs1 = self.feature_extractor(images=image, return_tensors="pt")
        inputs2 = self.feature_extractor(images=depth_image, return_tensors="pt")
        inputs = {}
        inputs['pixel_values'] = torch.cat((inputs1['pixel_values'], inputs2['pixel_values']), dim=1)
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

        # First, rescale logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image_size,
            mode='bilinear',
            align_corners=False
        )

        # Second, apply argmax on the class dimension
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        pred_img = TerrainSegFormer.get_seg_overlay(image, pred_seg)
        return pred_img, pred_seg

    @torch.inference_mode()
    def predict_new(self, image):
        with torch.device(self.DEVICE):
            pred_seg = self._internal_predict(image, image.size[::-1])[0]
            pred_img = TerrainSegFormer.get_seg_overlay(image, pred_seg.cpu(), alpha=0.2)
        return pred_img, pred_seg

    def predict_ds_metrics(self, pred_ds):
        pred_ds = {
            'pixel_values': np.stack([np.array(img) for img in pred_ds['pixel_values']]),
            'labels': np.stack([np.array(img) for img in pred_ds['labels']])
        }
        images = pred_ds['pixel_values']
        gt_segs = pred_ds['labels']
        pred_segs = np.array(self._internal_predict(images, images[0].shape[:-1]))
        gt_segs_array = np.stack([np.array(img) for img in gt_segs])

        metrics = self.metric._compute(
            predictions=pred_segs,
            references=gt_segs_array,
            num_labels=len(self.id2label),
            ignore_index=0,
            reduce_labels=self.feature_extractor.do_reduce_labels,
        )
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        metrics.update({f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics

    def predict_ds_metrics_wrapper(self, pred_ds, batch_size=8):
        # Create indices for batches
        num_samples = len(pred_ds['pixel_values'])
        indices = list(range(num_samples))
        np.random.shuffle(indices)  # shuffling may provide a more robust estimate of mean metrics
        batch_indices = [indices[i:i + batch_size] for i in range(0, num_samples, batch_size)]

        metrics_list = []
        batch_sizes = []

        pred_ds_array = {
            'pixel_values': np.stack([np.array(img) for img in pred_ds['pixel_values']]),
            'labels': np.stack([np.array(img) for img in pred_ds['labels']])
        }
        def get_batch(idx): return (pred_ds_array['pixel_values'][idx], pred_ds_array['labels'][idx])

        # Loop over all batches
        for bidx, batch in enumerate(batch_indices):
            print(f"Processing batch {bidx + 1}/{len(batch_indices)}...")
            pixel_values_batch, labels_batch = get_batch(batch)
            batch_ds = {
                'pixel_values': pixel_values_batch,
                'labels': labels_batch
            }
            print(f"Computing metrics for batch {bidx + 1}/{len(batch_indices)}...")
            metrics = self.predict_ds_metrics(batch_ds)
            metrics_list.append(metrics)
            batch_sizes.append(len(batch_ds['pixel_values']))

        # Compute the weighted mean metrics
        mean_metrics = {}
        for key in metrics_list[0].keys():
            mean_metrics[key] = np.sum([metrics[key] * size for metrics, size in zip(metrics_list, batch_sizes)]) / np.sum(batch_sizes)
        return mean_metrics

    @staticmethod
    def get_seg_overlay(image, seg, alpha=0.5):
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(TerrainSegFormer.DISTINCT_COLORS_51)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        # Show image + mask (i.e., basically transparency of overlay)
        img = np.array(image) * (1 - alpha) + color_seg * alpha
        img = img.astype(np.uint8)
        return img


def run_rosbag_inference_video(bagfile_path: str, image_topic: str):
    save_dirpath = "./test/bag_video_terrain"
    os.makedirs(save_dirpath, exist_ok=True)
    s = TerrainSegFormer()
    s.load_model_inference()
    bag = rosbag.Bag(bagfile_path, "r")
    bridge = CvBridge()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_terrain = None
    print(green(f"Processing number of messages: {bag.get_message_count(image_topic)}", ["bold"]))
    for i, (topic, msg, t) in enumerate(tqdm(bag.read_messages(topics=[image_topic]), desc="Processing Bag", total=bag.get_message_count(image_topic))):
        cv2_frame_np = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
        pil_img = Image.fromarray(cv2.cvtColor(cv2_frame_np, cv2.COLOR_BGR2RGB))
        pred_img_terrain, _ = s.predict_new(pil_img)
        pred_img_terrain = cv2.cvtColor(np.asarray(pred_img_terrain), cv2.COLOR_RGB2BGR)
        if writer_terrain is None:
            h, w = pred_img_terrain.shape[:2]
            writer_terrain = cv2.VideoWriter(os.path.join(save_dirpath, "terrain.mp4"), fourcc, 10, (w, h), True)
        writer_terrain.write(pred_img_terrain)
    writer_terrain.release()
    bag.close()


if __name__ == "__main__":
    s = TerrainSegFormer()
    s.load_model_inference()
    img = Image.open("test/000000.png")
    pred_img, pred_seg = s.predict_new(img)
    plt.imshow(pred_img)
    plt.show()

    # run_rosbag_inference_video("test/demonstrations/1.bag", "/zed2i/zed_node/left/image_rect_color/compressed")
