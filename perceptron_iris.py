import numpy as np
import sys


class Perceptron:
    """
    Perceptron class with methods to train and calculate accuracy.
    """
    def __init__(self, num_epochs=4, lr=1) -> None:
        # Number of training epochs
        self.num_epochs = num_epochs
        # Learning rate
        self.lr = lr
        # List of decision boundary vectors
        self.locilne_meje = []

    def train(self, X):
        """ 
        The method takes the matrix X as an argument

        """

        # Number of features
        num_features = X.shape[1]

        # Initialize a weight vector that is equal in length to the sample vector (vector of separation boundary coefficient coefficients)
        self.w = np.zeros(num_features)
        # We can also used random weight vector at the beginning
        #self.w = np.random.rand(num_features)

        # Calculate weights for every iteration
        for _ in range(self.num_epochs):
            # In every iteration go thorough all samples
            for i, sample in enumerate(X):
                 # Calculate value of decision boundary function
                d_x = np.dot(self.w, sample)
                if d_x <= 0:
                    # Correct weights vector
                    self.w += self.lr*sample

        return self

    def test(self, X_test):
        """ The method takes the matrix of test samples as an argument and returns a vector of labels of the classes of classified samples. """

        # Initialize the list of recognized patterns
        y_hat = []
        # For each sample in the test matrix
        for sample in X_test:
            # List of dot products
            ys = []
            # For each decision boundary we calculate dot product with the sample vector
            for meja in self.locilne_meje:
                ys.append(np.dot(meja, sample))

            # Calculate all three decision boundaries for between all three classes
            d_12 = ys[0]
            d_13 = ys[1]
            d_23 = ys[2]
            # Check which side of the dividing line the sample is located
            if d_12 > 0 and d_13 > 0:
                y_hat.append(1)
            elif d_12 < 0 and d_23 > 0:
                y_hat.append(2)
            elif d_13 < 0 and d_23 < 0:
                y_hat.append(3)
            else:
                # The sample cannot be sorted
                y_hat.append(0)

        return y_hat

    def score(self, y_test, y_hat):
        """ The method takes the vector of correct and the vector of recognized samples and returns the proportion of correctly recognized samples. """
        return np.mean(np.array(y_test) == np.array(y_hat))


def unpack_data(filename):
    """
    Takes the file name, returns list X and list y.
    """
    data = []
    try:
        with open("data/"+filename, 'r') as f:
            data = f.readlines()

    except IOError:
        print(f"Error opening {filename}")

    # Strip \n characters
    data = [item.strip() for item in data]

    # Create y vector and list od lists X
    y_seznam = []
    seznam = []
    X_seznam = []
    for indeks, element in enumerate(data):
        # Separate data with commas and obtain rows (list)
        seznam = element.split(",")
        # Attach class labels to the y list
        y_seznam.append(seznam[-1])
        # Remove class labels from the list
        seznam.remove(seznam[-1])
        # Attach the cleared list (line) to the new X_list.
        X_seznam.append(seznam)

    # Create list of lists for X matrix (string to float)
    for n_vrstica, vrstica in enumerate(X_seznam):
        for n_element, element in enumerate(vrstica):
            X_seznam[n_vrstica][n_element] = (float(element))

    # make a list of all class labels and use numbers 1,2 and 3 instead of labels
    flowers = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    y_nov = []
    for flower in y_seznam:
        y_nov.append(flowers.index(flower)+1)

    return X_seznam, y_nov


def create_pairs(X):
    """
    As an argument, the function takes an X matrix and returns three matrices of 2 sample classes. 
    """
    # For all three combinations we make pairs of two classes
    for i in range(3):
        if i == 0:
            # Take the first 100 samples (class 0 and 1)
            X_1 = X[:100]
        elif i == 1:
            # Take class 0 and class 2
            X_2_1 = X[:50, :]
            X_2_2 = X[100:, :]
            # Combine samples from both classes
            X_2 = np.concatenate((X_2_1, X_2_2), axis=0)
        elif i == 2:
            # Take class 1 and 2

            X_3 = X[50:]

    return X_1, X_2, X_3


def split_data(X):
    """
    As an argument, the function takes the matrix of samples X. As an output, it returns the matrices of the learning and test sets and the vector of the labels of the test set.

    """
    # Calculate the number of samples from the required proportions for the learning and test set
    a = 2*int(len(X)/3)
    b = int(len(X)-a)
    c = int(len(X)/2)

    # For the learning set, take 2/3 of the samples from U1
    X_train_1 = X[:a]
    # For the test set, take 1/3 of the samples from U1
    X_test_1 = X[b:c]

    # For the learning set, take 2/3 of the samples from U2
    X_train_2 = X[c:(c+b)]
    # For the test set, take 1/3 of the samples from U2
    X_test_2 = X[(c+b):]

    # Learning set
    X_train = np.concatenate((X_train_1, X_train_2), axis=0)
    # Test set
    X_test = np.concatenate((X_test_1, X_test_2), axis=0)

    # Extend the learning samples by 1
    X_train = np.insert(X_train, len(X[0]), 1.0, axis=1)
    U1 = np.array(X_train[:int(len(X)/2)])

    # Multiply the samples from U2 by -1
    U2 = np.multiply(X_train[int(len(X)/2):], -1.0)

    # Assemble an extended learning set for 1 dimension
    X_train = np.concatenate((U1, U2),)

    # Extend the test samples by 1
    X_test = np.insert(X_test, len(X[0]), 1.0, axis=1)

    # Create a vector of test sample labels
    y_test = []
    for ind, sample in enumerate(X_test):
        # Up to half we have samples from the first class, then from the second
        if ind < len(X_test)/2:
            y_test.append(1)
        else:
            y_test.append(-1)

    return X_train, X_test, y_test


def test_labels(y):
    """ The function takes a vector of class y labels as input. Returns a vector of test sample labels as the output. """
    
    # Prepare sample labels
    y_1 = y[34:50]
    y_2 = y[84:100]
    y_3 = y[134:]

    # Compile labels for all 3 classes
    y_test = np.concatenate((y_1, y_2, y_3),)

    return y_test


def test_data(X_test_1, X_test_3):
    """ 
    The function takes as an input a matrix with test samples from classes 1 and 2 and a matrix with test samples from classes 1 and 3.
    As a result, it returns a test matrix with 16 test samples from classes 1, 2, and 3. The output matrix is ​​used in testing trained boundaries.
     """
    # A complete combination of Class 1 and 2 test samples is used
    T_1 = X_test_1
    # Only Class 3 test samples are used
    T_2 = X_test_3[int(len(X_test_3)/2):]
    # Combine test samples from all three classes
    X_test = np.concatenate((T_1, T_2),)

    return X_test


if __name__ == "__main__":

    try:
        # Pass the file name, number of iterations and learning rate as an arguments
        filename = sys.argv[1]
        n_epochs = int(sys.argv[2])
        l_r = float(sys.argv[3])
    except IndexError:
        print("Usage of the script: python perceptron_simple.py <filename> <number of iterations> <learning rate>")
        sys.exit(1)

    # From the data we get the lists X and y
    X, y = unpack_data(filename)

    # From the lists we make the matrix X and the vector y
    X = np.array(X)
    y = np.array(y)

    # Make pairs for all 3 combinations
    X_1, X_2, X_3 = create_pairs(X)

    # Make an instance of the perceptron, define the number of iterations and the learning rate.
    perceptron = Perceptron(num_epochs=n_epochs, lr=l_r)

    locilne_meje = []
    # Perform learning for each combination
    for iter in range(3):

        if iter == 0:
            # Obtain a learning and test set and a vector of test sample labels
            X_train, X_test_1, y_test_1 = split_data(X_1)

        elif iter == 1:
            X_train, X_test_2, y_test_2 = split_data(X_2)

        elif iter == 2:
            X_train, X_test_3, y_test_3 = split_data(X_3)

        # Perform separation boundary learning
        perceptron.train(X_train)

        # Prints the boundary boundary vector between the sample classes
        print(
            f"Vector of coefficients of the separation boundary {iter}: {perceptron.w}")

        # Save the boundary vector in the list of vectors
        perceptron.locilne_meje.append(perceptron.w)

    
    # Prepare a matrix with test data
    # 1/3 from each class       
    X_test = test_data(X_test_1, X_test_3)

    # Prepare a vector of test sample labels
    y_test = test_labels(y)

    # Calculate the vector of class labels on the test samples
    y_hat = perceptron.test(X_test)

    # Calculate accuracy
    score = perceptron.score(y_test, y_hat)

    print(f"Correct labels: {y_test}")
    print(f"Predicted labels: {y_hat}")
    print(f"Score:\t{score*100:.2f}%")
    print()
