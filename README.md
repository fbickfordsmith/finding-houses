# Finding houses in satellite images
Freddie Bickford Smith\
December 2019


## Results
Below is a trained network's predictions on an image section not seen during training. Predictions for the whole input image (ie, both training and validation sections) are at `predictions/train_valid_[image-style].png`.

![](/predictions/valid_comparison.png)


## Repository guide

File|What it does
-|-
`task.pdf`|Defines task
`train_predict_visualise.ipynb`|Shows workflow
`train.py`|Trains network to find houses
`predict.py`|Computes predictions with trained network
`visualise.py`|Visualises training and predictions
`helper_functions.py`|Defines functions used in other `.py` files
`requirements.txt`|Shows package versions used (auto-exported by pip in Colab)


## Discussion of main decisions

- Used 256x256px frames as input to the network. Could have used smaller frames but reasoned that a smaller frame contains less contextual information, thus probably making the task harder for the network.

- Reserved a 256px-tall strip of the training images for validation. Decided that this was the minimum amount of training data that could provide some meaningful measure of ability to generalise to unseen data.

- Used some basic data augmentation (random sampling of frame location; random flips of frames) to increase the effective size of the dataset and thus mitigate overfitting a bit. No other preprocessing used, but could be worth trying.

- Trained on 40,000 sampled frames because that corresponded with the amount of available memory on Colab. With less memory available, could resample the training frames periodically (eg, every 10 epochs).

- Used binary cross-entropy loss with `from_logits=True` because this is a binary-classification problem and the network's output is unbounded. Might be a good idea to use a weighted cross-entropy function because there is a class-imbalance issue with this data (pixels not belonging to a building are much more numerous than pixels belonging to a building).
