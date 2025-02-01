import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from emonet.models import EmoNet
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from typing import Optional, Dict, Union, List
import torch.nn.functional as F

class EmonetInferenceModel(nn.Module):
    '''
    Detects face in the image and predicts the emotion of the face. Predicts 5 emotions
    (Neutral, Happy, Sad, Angry, Surprised) and valence and arousal values.
    '''
    EMOTION_CLASSES = {
        0: "Neutral",
        1: "Happy",
        2: "Sad",
        3: "Surprise",
        4: "Fear",
    }
    def __init__(self, device = 'cuda'):

        super(EmonetInferenceModel, self).__init__()
        self.device = device
        self.image_size = 256

        # Loading the model
        state_dict_path = Path(__file__).parent/"pretrained"/ f"emonet_5.pth"
        state_dict = torch.load(str(state_dict_path), map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.emonet = EmoNet(n_expression=5).to(device)
        self.emonet.load_state_dict(state_dict, strict=False)
        self.emonet.eval()
        self.emonet.to(device)

        # Load face detector
        self.face_detector = SFDDetector(device)


    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> List[Optional[Dict[str, Union[str, float]]]]:
        '''
        Returns a list of dictionaries containing the emotion and valence and arousal values of the detected faces.
        if no face is detected, returns None as list item. Runs on first face if multiple faces are detected.

        :param images: BCHW [-1, 1] RGB tensor
        '''

        images_rgb255 = (images + 1) * 127.5
        batch_faces = self.face_detector.detect_from_batch(images_rgb255)

        results = []
        for image_idx, faces in enumerate(batch_faces):
            # If no face is detected, append None to the results
            if len(faces) == 0:
                results.append(None)
            else:
                bbox = np.array(faces[0]).astype(np.int32)
                face = images[image_idx, :, bbox[1]:bbox[3], bbox[0]:bbox[2]].unsqueeze(0)
                face = F.interpolate(face, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
                face = (face*0.5 + 0.5).to(self.device)  # [-1, 1] to [0, 1]
                output = self.emonet(face)
                results.append({
                    "emotion_logits": output['expression'].cpu().numpy(),
                    "valence": output['valence'].item(),
                    "arousal": output['arousal'].item()
                })

        return results






if __name__ == "__main__":
    import torchvision
    model = EmonetInferenceModel()
    #model.face_detector.detect_from_image("/hdd/ffhq/images1024x1024/02000/02002.png")
    image = torchvision.io.decode_image("/hdd/ffhq/images1024x1024/02000/02002.png").unsqueeze(0)
    image = image/127.5 - 1
    model(image)