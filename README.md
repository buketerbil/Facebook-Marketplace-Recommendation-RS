# Facebook Marketplace Recommendation Ranking System
## 1. Cleaning tabular data

From the Amazon Web Services (AWS) cloud account that was provided to me during my training, I obtained tabular dataset (csv file) that of Facebook Marketplace products, including information about the listing, price, categories, locations and descriptions. 

There were several entries with complete rows of missing data,  entries with no locations or descriptions in the tabular data. These data might influence the learning process and consequently accuracy of my model during training, 
<img width="974" alt="Screenshot 2022-10-06 at 18 22 32" src="https://user-images.githubusercontent.com/102605064/194378445-7805b44f-8c89-4596-83d1-c6b88569791e.png">

- I cleaned them with a simple code by dropping the rows containing NaN values.
- This was done by creating a separate py file to clean tabular data, using pandas library.

## 2. Making sure all images have the same size, number of channels and quality

For the image classification models later, it was necessary to make sure the image dataset is consistent. 
<img width="1345" alt="Screenshot 2022-10-06 at 18 19 53" src="https://user-images.githubusercontent.com/102605064/194377924-a986f6d5-4696-4835-a782-6ec66af64fc6.png">

- Using Python Imaging Library (PIL), I used a function that takes in an image and sets its size to 400x400, and quality to 100, as JPEG
- I wrote a code that iterates through the file containing images, applies the resize_image function and saves the resized images to a new folder named ‘resized images’

## 3. First regression model

Regression wasn’t going to be used in the final system of my project (since it focuses on learning product embeddings by pre-training on classification tasks), nonetheless I created a simple  regression model to predict the price of the products by taking in the features: product name, product description and location.

I created a function to

- Vectorise product name, description and location by using TF-IDF vectoriser to the text data
- Pass these features as X, and the price values that were stripped off of ‘£’s (to type them as floats) as Y
- Fit these onto a scikit learn Linear Regression model, and predict the first 10 examples

## 4. First simple classification model

I created a classification model that predicts the category of each product.

- I first merged the products dataframe and images data frame (that contains product ids and ids of the images) on the product id feature
- I created a pandas data frame of the images’ paths from the resized images folder (using glob), to then be able to strip the image names off of the ‘.jpg’ ends - and only have their ID names left. Then I merged it with the previously merged data frame on the features id and category
- I made sure to get rid of the extra descriptions at the end of the categories to have only the most general category
-


## 5. Image Classification with CNNs in PyTorch 

The first ulticlass image classification model was built in PyTorch. However, the Convolutional Neural Network architecture that I built was improved by fine-tuning a pre-trained RESNET-50 model instead.
