# Image Difference Detector — NMTronics NEST Assignment

## Structure

```
code/
  generate\\\_samples.py   # creates 5 synthetic near-identical PCB image pairs
  diff\\\_detector.py       # core CV pipeline: SSIM -> Otsu threshold -> morphology -> contours -> bboxes
  app.py                  # Flask API + browser UI wrapping the pipeline (service layer)
  requirements.txt
raw\\\_samples/              # the 5+ generated input pairs (pair\\\_N\\\_A.png / pair\\\_N\\\_B.png)
results/                   # matplotlib output images with ROI bounding boxes + summary.json
documentation/             # Word doc covering architecture, make diagrams, edge deployment, AI usage
```



## Setup

```bash
#Create \& Activate Virtual Environment and than install the requirements.txt
pip install -r code/requirements.txt
```



## Run (batch mode — reproduces all results in /results)

```bash
cd code
python generate\\\_samples.py    # regenerate raw sample pairs (optional, already included)
python diff\\\_detector.py         # runs detection on all pairs, saves results/\\\*.png + summary.json
```



## Run (interactive web demo)

```bash
cd code
python app.py
# open http://localhost:5000, upload any two images, get instant bbox visualization
```



## API (for programmatic use)

```
POST /detect
  form-data: image\\\_a=<file>, image\\\_b=<file>
  header: Accept: application/json
  -> { ssim\\\_score, num\\\_regions\\\_detected, boxes\\\_xywh, result\\\_image\\\_base64 }
```

