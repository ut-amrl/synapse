import os
import sys
import cv2
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from third_party.depthany2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class DepthAny2:
    def __init__(self, device=None, model_input_size=518, max_depth=15):
        if device is not None:
            self.DEVICE = torch.device(device)
        else:
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.DEVICE == 'cuda':
            assert torch.cuda.is_available()
        self.model_input_size = model_input_size
        self.max_depth = max_depth
        self.setup_()

    def setup_(self):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        encoder = 'vitl'
        dataset = 'hypersim'
        self.depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': self.max_depth})
        self.depth_model.load_state_dict(torch.load(f'third_party/depthany2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=self.DEVICE))
        self.depth_model.eval()
        self.depth_model.to(self.DEVICE)

    @torch.inference_mode()
    def predict(self, cv2_img, max_depth=None):
        depth_arr = self.depth_model.infer_image(cv2_img, self.model_input_size, max_depth)
        return depth_arr


if __name__ == "__main__":
    image_path = "/home/dynamo/Downloads/merobot.png"

    print("Testing Depth")
    cv2_img = cv2.imread(image_path)
    depth_obj = DepthAny2()
    depth_arr = depth_obj.predict(cv2_img)
    cv2.imshow("Depth", (depth_arr * 4000).astype(np.uint16))  # for better vis, multiply by 4000
    cv2.waitKey(0)
    cv2.destroyAllWindows()
