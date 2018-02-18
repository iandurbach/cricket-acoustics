# research-code
All the files contained in this repo were used for my research: Machine learning methods for individual acoustic recognition in a species of field cricket.

List of files and their description
1. segmentation_feature_extraction.R file contains functions that were used to segment the acoustic recordings (into chirp and syllable level) and also extract the MFCC features, temporal features as well as raw sample extraction. The number of MFCC coefficients per frame is a parameter that needs to be chosen (13 was used in this research). The R library function that reads in the audio file expects a file in WAV format. 

There are multiple .py files in this repo.

2. functions.py file contains function definitions that are used to perform: some downsampling of the acoustic samples and classifiers for the actual classification. There are no function calls in this file.

3. mfcc_models.py does some preprocessing (standardising, frame truncation and padding to 200 frames) of the MFCC features/matrices as well as function calls for both NN and RF classifiers to classify the MFCC features. NN classifier takes in an input tensor of dimension n x 200 x 13, and n x k one-hot-encoded binary matrix of class labels, where n is the number of chirp-level training MFCC matrices, k is the total number of classes. The RF classifiers takes in an input matrix of m x 13, and and m x 1 column matrix of class labels where m is the number of frame-level training MFCC vectors.

4. raw_samples_models.py does some preprocessing (filtering, downsampling, and sequence truncation and padding to 3000 samples) of the raw acoustic samples as well as function calls for the NN models that were used to classify these inputs. Classifier takes in an input tensor of dimension n x 3000 x 1, and n x k one-hot-encoded binary matrix of class labels, where n is the number of chirp-level training sequences, k is the total number of classes.

5. temp_feat_models.py does some preprocessing (standardising) of the temporal features as well as function calls for the RF models that were used to classify individuals based on these features. Classifier takes in n x 2 matrix of features, and n x 1 column matrix of class labels, where n is the number of training chirp-level training examples. 

For the mfcc_models.py and raw_samples_models.py files, the numbers 200 and 3000 were dependent on the nature of the data that was used in this research. Further exploration would be required to set the right values for a different dataset.












