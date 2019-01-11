# Keras-Med-IO
Providing fast and easy IO for on-the-fly and local preprocessing in Keras.
## Main Utilities
### Generators
* __BaseGenerator__
  * The main simple generator to inherit from that's based on `keras.utils.Sequence`. It is thread-safe and intuitive to use.
* __PosRandomPatchGenerator__
  * Generates windows on the fly with data augmentation based on the configuration and the number of positive class images (__Channels last__)

### Preprocessing
* `io_func.py`
  * One-Hot Encoding
  * Normalization
  * Resampling
  * get_list_IDs
* __Window/Patch Extraction__
  * `patch.py` and `patch_utils`
    * `patch.py` contains the object-oriented patch extractors (`PatchExtractor`, `PosRandomPatchExtractor`)for code reusue through inheritance
    * `patch_utils.py` contains the function patch extraction code from [ellisdg's 3DUnetCNN repository](https://github.com/ellisdg/3DUnetCNN)
* __Data Augmentation__
  * From [MIC-DKFZ's batchgenerators](https://github.com/MIC-DKFZ/batchgenerators), but it is _channels first_

## To-Do
1) Need to move data_aug to utilize the data aug from [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
2) Dataset from generator
  * .from_generator
  * or have a function to parse a dataset of filenames
3) Refactoring
  * have only one generator with one function that extracts patches of all types
  * refactor the deprecated generators with cleaner code
