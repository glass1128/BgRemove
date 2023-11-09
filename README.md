# Background Removing for Passport Applications
This project is about providing a background removing feature by AI & Deep learning, the main features:
- Transparent background
- White background
- Passport Photo (in case of human photos)

The code works on all types of images not only humans

## Results
![result1](images/2.png)
![result1](images/1.png)

## Install
```bash
pip install -r requirements.txt
```

## Run on single image
```bash
python3 remove_background.py --image "PATH_TO_IMAGE" --transparent --white_background --passport --visualize

# --transparent --white_background --passport are boolean options, so you can remove anyone of them to not include it.

# Replace "PATH_TO_IMAGE" by the image path on your disk
```
## Run on folder of images
```bash
python3 remove_background.py --folder "PATH_TO_FOLDER" --transparent --white_background --passport --visualize

# --transparent --white_background --passport are boolean options, so you can remove anyone of them to not include it.

# Replace "PATH_TO_FOLDER" by the folder path that contains many images
```
All past commands just visualize the results, to save the results on the disk add the `--save_path` option

```bash
python3 remove_background.py --image "PATH_TO_IMAGE" --transparent --white_background --passport --visualize --save_path "PATH_TO_SAVE_RESULTS"

# Note you can remove the --visualize option for automation
```

inside the `save_path` that you givin, will be created 3 folders [`transparent`, `white_background`, `passport`] that saves the result images (with the same name of input images) in each corresponding folder

## U2Net vs Rough Image Matting
the algorithm runs both `U2net` for background removal and `Rough Image Matthing` for human segmentation, based on face detection it selects one of the models
To alawys run `Rough Image Matting` to avoid issues use flag `--always_matting`

## Positioning of passport
in `passport.py` you will find these variables on the top
```python
WIDTH_FACTOR = 0.6
HEIGHT_FACTOR_TOP = 0.5
HEIGHT_FACTOR_BOTTOM = 0.6
```
adjust them for the positioning you want. higher number means larger area in that direction relative to face, for example when you increase `HEIGHT_FACTOR_TOP` it means larger area on top of the face will be included in the croped image

python C:\Users\Administrator\pyScript\BackgroundRemoval\SemanticGuidedHumanMatting\remove_background.py --image "C:\Websites\CentredTherapies\img\carolyn.jpg" --passport --save_path "C:\Websites\SAASTemplate\processed"