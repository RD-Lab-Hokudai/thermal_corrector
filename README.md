## Thermal image corrector

A distortion corrector for Optris-PI 640.

### Features

- Estimate distortion parameters from single aligned calibration image.
- Evaluate the accuracy of distortion correction.

### Requirements

- Python 3.6+
- opencv-contrib-python 4.5.1

### How to use

1. Create calibration image

Refer to my master thesis.

This tool requires aligned calibration image.

For Optris-PI 640, sample/calib_pi_640.png is suitable as a calibration image.

2. Extract points from calibration image

```
$ python3 center_extractor.py <calibration_image_path>
```

ex)

```
$ python3 center_extractor.py sample/calib_pi_640.png
```

This script creates an image file named as 'points.png'.

3. Estimate distortion parameters

```
$ python3 corrector.py <points_image_path>
```

To work this, you must have a points image file.

For Optris-PI 640, sample/points_pi_640.png is suitable as a points image.

ex)

```
$ python3 corrector.py sample/points_pi_640.png
```

This script estimates distortion parameters and variance (for evaluation).
