# neuralnets
neural networks in pytorch using pytroch and tensorflow

@c-ateba
c-ateba Add files via upload

A neural network model to classify 3 types of iris flowers (setosa, versicolor, virginica).
Input size is 4 and output size is 3. However we need just one output hence the argmax applied to the 3 output columns. The model is therefore producing the probability that the input represents one of the 3 classes of Iris. 
To be remembered:
np.argmax(prediction.detach(),axis=1). It took a while to understand that ArgMax will return the class id of the 3 columns (0,1, or 2) corresponding to the class with highest output probability.
