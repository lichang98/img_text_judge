# img_text_judge
Judge whether a scene image contains text\
The project contains `hog_desc.py` which extracts `HOG` feature from an image. The `HOG` feature will be feed into `Random Forest Classifier` to judge whether a scene image contains text.\
The fundmental principals of `HOG` feature extraction algorithm, you can search related blogs for details.\
The project contains data for training and validation. The data are saved with `pickle`,
and compressed with `zip`. Unzip data before running the algorithm.
