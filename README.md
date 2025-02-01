# Estimation of continuous valence and arousal levels from faces in naturalistic conditions, Nature Machine Intelligence 2021

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#Citation).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find a full-text, view only, version of the paper [here](https://rdcu.be/cdnWi).

The full article is available on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

## Youtube Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=J8Skph65ghM">Automatic emotion analysis from faces in-the-wild
  <br>
  <img src="https://img.youtube.com/vi/J8Skph65ghM/0.jpg"></a>
</p>

## Installation
```bash
pip install git+https://github.com/puttarwar/emonet.git

```

## Usage
n_classes: number of classes for emotion classification (default: 5, i.e., neutral, happy, sad, angry, surprised)

### Testing on a video
Uses the first detected face in a video and predicts their emotion, valence, arousal and facial landmarks.

```python
from emonet.demo import run_on_video
run_on_video(n_classes=5, video_path='path/to/video.mp4', output_path='path/to/output.mp4')
```

### Testing on an image
Uses the first detected face in a video and predicts their emotion, valence, arousal and facial landmarks.
    
```python

from emonet.demo import run_on_image
run_on_image(n_classes=5, image_path='path/to/image.jpg', output_path='path/to/output.jpg')
```

### Inference on a tensor