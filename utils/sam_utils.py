import torch
import torchvision
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator



class SAM:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = self.sam_load()

    def sam_load(self):
        sam_checkpoint = "./load/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        predictor = SamPredictor(sam)

        print('[INFO] Successfully load SAM Predictor')
        return predictor
    
    def get_mask(self, image, input_point, input_label):
        ## input: image [h,w,3] numpy uint8
        # if isinstance(image, torch.Tensor):
        #     img_numpy = image.permute(0,2,3,1).squeeze(0).cpu().detach().numpy() #[h,w,3]
        #     img_numpy = (img_numpy.copy() * 255).astype(np.uint8)
        # else:
        #     img_numpy = image
        
        input_point = np.array(input_point)
        input_label = np.array(input_label)
        
        img_numpy = image
        self.predictor.set_image(img_numpy)
        
        if len(input_label) > 0:

            masks, scores, logits = self.predictor.predict(
                                                        point_coords=input_point,
                                                        point_labels=input_label,
                                                        multimask_output=True)

            mask_input = logits[np.argmax(scores), :, :] 

            mask, _, _ = self.predictor.predict(
                                                point_coords=input_point,
                                                point_labels=input_label,
                                                mask_input=mask_input[None, :, :],
                                                multimask_output=False)
            
            mask = (mask[0] * 255).astype(np.uint8)
        else:
            mask = None
        
        return mask  #[h,w]