A list of best practices for neural networks, machine learning and data science in general culled from personal experience and other people's best practices lists

# General

## Hodgepodge

- You can set a value of something in place by using an underscore at the end of the name, e.g., instead of writing `<var>.weight.data = uniform(-5,5)` you can write `<var>.weight.data.uniform_(-5,5)`
- To get the size (resolution) of an image you can use the linux command `file <image_name>`

# Data

## General rules
- Always look at your data, both as raw values and as plots, before doing any serious munging, analysis, or model building
   -  Remove outliers which make sense and there is no other variable to capture those outliers
- Do as much of your work as you can on a small sample of the data

## Splitting Data

There is no one right answer, but assuming you are not data limited it seems that a train-test split of anything between 60%-40% to 80%-20% is fine.  As an aside, the dafult in sklearn is 75%-25%.  I tend to like 60%-40% and then further split my test data into 20% test and 20% validation.

If your dataset is small then 20% of the data for validation might not be enough. If the validation score fluctuates significantly every time you retrain your model then your validation set is probably too small.  But this also depends on the degree of accuracy you care about, as the greater accuracy required will correlate to the size of the validation set.

### Feature Engineering
- Expand datetime variables in order to extract features useful for modeling, e.g., day of week, weekday vs weekend, month, year, holiday ...

# Neural Networks

## Classification vs regression

- Activation functions
  - Classification, Binary: log-softmax
  - Classification, Multi-label: sigmoid
  - Regression: linear
- Loss function
  - Classification: cross-entropy
  - Regression: RMSE

## Embeddings

A rule of thumb for the size of your embeddings is either 50 or half the number of categories plus 1, taking the smaller of those two options.

## Activation Functions
- Theoretically softmax and logsoftmax are scaled version of each other, but empirically logsoftmax is better
- Sigmoid instead of softmax for multi-label classification
- In hidden state to hidden state transition weight matrices tanh is used

## Pre-built models

- In pytorch (assuming you have already installed torchvision) To view the architecture of any stndard pre-built model, e.g., resnet34, vgg16, you can simply call `torchvision.models.<architecture>()`

# Training

## Overfitting

### what are different ways to prevent overfitting

1. For neural networks, intially training the model on small data sizes/images for the first few epocs and then switching to bigger data sizes/images
  - Fastai has this built in using the `<learn_function>.set_data()` function.

# Fast AI

## Building neural networks

### Structured Data

Assuming you have finished your feature engineering and data munging, converted categorical variables to numbers, and split off your target variable, here are the steps you take to build your neural network .

1. Create a list of all the categorical and continuous variables in your pandas dataframe
1. Create a list of rows in your training dataset that you want to be in your validation dataset
1. Call `<data_function> = ColumnarMOdelData.from_data_frame()`
1. Create a list of embedding sizes, one tuple for each categorical variable
1. Call `<learn_function> = <data_function>.get_learner()`
1. Call `<learn_function>.fit()`



### Leaning rates

- Use learning rate finder and select a learning where convergence of loss is steep
- When using a pretrained model on some dataset like imagenet, you need to use different learning rates when you are using that model for any new dataset. The initial layers need a smaller learning rate, and the deeper layers need a comparatively larger learning rate.
   - The magnitude of the learning rates should vary depending on the similarity between the dataset on which the model was originally trained and the new dataset
- Cosine annealing
- SGD with restarts

### Training

- bn_freeze = True: In case you are using a deeper network anything greater than resnet34 (like resnext50), bn_freeze = True should be used when you unfreeze and your new dataset is very similar to the original data used in pretrained model
- On the other hand, when the new dataset is not similar to the original dataset, we start with smaller size 64x64, fit in freezed state, unfreeze and fit. And repeat the process with 128x128 and 256x256




