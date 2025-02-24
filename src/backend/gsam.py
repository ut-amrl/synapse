import cv2
import torch
import torchvision
import numpy as np
import supervision as sv
from segment_anything import sam_hq_model_registry, SamPredictor
from groundingdino.util.inference import Model


class GSAM:
    def __init__(self, box_threshold=0.25, text_threshold=0.25, nms_threshold=0.6, ann_thickness=2, ann_text_scale=0.3, ann_text_thickness=1, ann_text_padding=5, device=None):
        if device is not None:
            self.DEVICE = torch.device(device)
        else:
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.DEVICE == 'cuda':
            assert torch.cuda.is_available()
        self.GROUNDING_DINO_CONFIG_PATH = "gsam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.GROUNDING_DINO_CHECKPOINT_PATH = "gsam/weights/groundingdino_swint_ogc.pth"
        self.grounding_dino_model = Model(model_config_path=self.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=self.GROUNDING_DINO_CHECKPOINT_PATH, device=self.DEVICE)
        self.SAM_ENCODER_VERSION = "vit_h"
        self.SAM_CHECKPOINT_PATH = f"gsam/weights/sam_hq_{self.SAM_ENCODER_VERSION}.pth"
        self.sam = sam_hq_model_registry[self.SAM_ENCODER_VERSION](checkpoint=self.SAM_CHECKPOINT_PATH)
        self.sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.NMS_THRESHOLD = nms_threshold
        self.box_annotator = sv.BoxAnnotator(
            thickness=ann_thickness,
            text_scale=ann_text_scale,
            text_thickness=ann_text_thickness,
            text_padding=ann_text_padding
        )
        self.mask_annotator = sv.MaskAnnotator()

    @torch.inference_mode()
    def predict_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO on image.
        img: A x B x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels
            detections:
                - xyxy: (N, 4) boxes (float pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
        """
        if box_threshold is None:
            box_threshold = self.BOX_THRESHOLD
        if text_threshold is None:
            text_threshold = self.TEXT_THRESHOLD
        if nms_threshold is None:
            nms_threshold = self.NMS_THRESHOLD
        detections = self.grounding_dino_model.predict_with_classes(
            image=img,
            classes=text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        # print(f"Detected {len(detections.xyxy)} boxes")
        if do_nms:
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold
            ).numpy().tolist()
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            # print(f"After NMS: {len(detections.xyxy)} boxes")
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = self.box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)
        return annotated_frame, detections

    @staticmethod
    @torch.inference_mode()
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    @staticmethod
    @torch.inference_mode()
    def get_per_class_mask(img, masks, class_ids, num_classes):
        """
        Create a per-class segmentation mask.
        Parameters:
            masks: N x H x W array, where N is the number of masks
            class_ids: (N,) array of corresponding class ids
            num_classes: Total number of classes, C
        Returns:
            per_class_mask: C x H x W array
        """
        H, W = img.shape[0], img.shape[1]
        per_class_mask = np.zeros((num_classes, H, W), dtype=bool)
        if len(masks) == 0:
            return per_class_mask
        for i in range(num_classes):
            class_idx = np.where(class_ids == i)[0]
            if class_idx.size > 0:
                per_class_mask[i] = np.any(masks[class_idx], axis=0)
        return per_class_mask

    @torch.inference_mode()
    def predict_and_segment_on_image(self, img, text_prompts, do_nms=True, box_threshold=None, text_threshold=None, nms_threshold=None):
        """
        Performs zero-shot object detection using grounding DINO and segmentation using HQ-SAM on image.
        img: H x W x 3 np cv2 BGR image
        text_prompts: list of text prompts / classes to predict
        Returns:
            annotated_frame: cv2 BGR annotated image with boxes and labels and segment masks
            detections: If there are N detections,
                - xyxy: (N, 4) boxes (int pixel locs) in xyxy format
                - confidence: (N, ) confidence scores
                - class_id: (N, ) class ids, i.e., idx of text_prompts
                - mask: (N, H, W) boolean segmentation masks, i.e., True at locations belonging to corresponding class
        """
        _, detections = self.predict_on_image(img, text_prompts, do_nms=do_nms, box_threshold=box_threshold, text_threshold=text_threshold, nms_threshold=nms_threshold)
        detections.mask = GSAM.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        labels = [
            f"{text_prompts[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = self.mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        detections.xyxy = detections.xyxy.astype(np.int32).reshape((-1, 4))
        return annotated_image, detections, GSAM.get_per_class_mask(img, detections.mask, detections.class_id, len(text_prompts))


if __name__ == "__main__":
    image_path = "temp.png"

    print("Testing Grounded SAM")
    cv2_img = cv2.imread(image_path)
    gsam_obj = GSAM()
    ann_img, *_ = gsam_obj.predict_and_segment_on_image(img=cv2_img, text_prompts=["door"])
    cv2.imshow("Grounded SAM", ann_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
