# Transforms
This is module of transforms that can be composed and utilized for the generators in this repository. It is primarily based on
[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) and their augmentations. However, since all of the augmentations for that repository are for `channels_first` data, we need to build wrappers for them so that they compatible with the current
generators (which are for `channels_last`).

__Goal:__
    * to develop batchgenerator-like transforms
__Features:__
    * Easily composed
    * OOP
    * easily integrated into the generator interface
    * Works with channels_last data
__OR__ [`transforms_generator.py`] <br>
1) Make a generator that generates the data and just does the conversion of channels_last to first for the transform
      * Needs to make data dict after loading image
      * converts channel format
      * passes it to transform
      * get the resulting dict, convert back, and return (data_dict['data'], data_dict['seg'])

__(outdated) To-Do List:__ <br>
* AbstractTransform
* ComposeTransform
* Individual Transforms for tuples (x,y)
    * Make compatible for multiple inputs as well
* Have a transform to convert data between channels_first and channels_last so that it's fluid
