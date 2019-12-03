# Hand-objects-contact-probability

Predict the probability of hand-object contact probability

Predicting the probability of contact between objects, we first use yolov3 as the detector, and social-lstm to predict the moving path. Finally, we adopt the convolutional neural networks to learn the potential relation between the position information of objects, and output the contact probability.
