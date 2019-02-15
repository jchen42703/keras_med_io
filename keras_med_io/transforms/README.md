# Transforms
This is a module of generators that utilize data augmentation operations from [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) and their transforms.

__Features:__
    * Easily composed
    * OOP
    * easily integrated into the generator interface
    * Works with channels_last data

__`transforms_generator.py`__ <br>
    * Example of using this generator with `batchgenerators` is located in the `examples` folder.
    * __How it works with the `batchgenerators` transforms:__
      * __Original Problem__: the transforms only work with `channels_first` data, but this repository focuses on `channels_last`
      solutions.
      * makes data dict after loading image
      * converts channel format to `channels_first`
      * passes it to transform
      * get the resulting dict, convert back to `channels_last`, and return (data_dict['data'], data_dict['seg'])
