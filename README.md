# Makeup Transfer Using no Deep Learning components

This is the side project for [LADN](https://github.com/wangguanzhi/LADN), and provide a pre-processing pipeline for in-the-wild before and after makeup face images. The images need to be annonated as before-makeup or after-makeup.

The major functions used is facial landmarks detection, Delauney triangulation, image warping and Possion blending. Some heuristic are also used to process the results to ease the color bleeding effect.


## Setup

This project involves 3 type of face detector (dlib, Stasm and Face++ Detect API)

* Register for the Face++ service, get a API key (This should be free of charge). Update the `FACEPPAPI_KEY` and `FACEPPAPI_SCRECT` in `src/settings_template.py` and change the name of the file to `src/settings.py`
* Download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, extract the file and put it in the `data` folder.
* Install the dependencies (using Anaconda)
```
conda env create -f environment.yml
conda activate makeup-transfer
```

In case of error of being unable to find OpenCV header, please try to include the opencv header directory in the environment, e.g `export "CPATH=/path/to/anaconda3/envs/makeup-transfer/include"`.

## Sample Usage

For the makeup transfer, the main pipeline is to crop the face region of the an after makeup image, warp it towards the the face of a before makeup image, and use poisson blending (`cv2.seamlessClone`) to blend the two images together.

For this purpose, 3 type of face detector (dlib, Stasm and Face++ Detect API) are used, where dlib face detector are only used to detect faces and align them to the upright rotation. If you have other pre-processing pipelin for alignment and cropping, please feel free to skip the step 1 and 2 below. The dataset provided in [LADN](https://github.com/wangguanzhi/LADN) is already after alignment and cropping.

A sample dataset are provided in `data/raw_sample/`.

1. Given a set of images in the wild in folder `data/raw/` where before- and after-makeup images are put in `before/` and `after/` subfolder respectively, the following command copy and rename the images, and filter out those images that cannot be detected by `Stasm`.
```
python src/handle_dataset.py --type filter --input data/raw/ --output data/filter/ --rename --detector stasm
```
2. Then detect the faces in the images, align them to the upright rotation and crop the image region centered at the face.
```
python src/handle_dataset.py --type crop --input data/filter/ --output data/crop
```
3. Use Face++ Detect API to get accurate face landmarks, and store them into a file.
```
python src/handle_dataset.py --type landmark --input data/crop/ --output data/crop/landmark.pk
```
4. Crop, warp and blend the after makeup face onto the before makeup faces. This step requires the images in `data/crop/` to be separated into `before/` and `after/` subfolders.
```
python src/handle_dataset.py --type blend --input data/crop/ --output data/crop/blend/ \
  --landmark_input data/crop/landmark.pk --keep_eye_mouth --include_forehead --adjust_color --adjust_lighting
```

## Citation

If you used this project in your research, please cite:
```
@inproceedings{gu2019ladn,
  title={Ladn: Local adversarial disentangling network for facial makeup and de-makeup},
  author={Gu, Qiao and Wang, Guanzhi and Chiu, Mang Tik and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={10481--10490},
  year={2019}
}
```
