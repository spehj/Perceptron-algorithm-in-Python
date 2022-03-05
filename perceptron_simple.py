import numpy as np
import matplotlib.pyplot as plt
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

    def train(self, X):

        # Number of features
        num_features = X.shape[1]
        # Initialize a weight vector equal to the length of the feature vector extended by 1 (separation boundary coefficient vector)
        self.w = np.zeros(num_features+1)
        # We can also start at the random weight vector
        #self.w = np.random.rand(num_features+1)

        # Extend features for 1 dimension
        X = np.insert(X, len(X[0]), 1.0, axis=1)
        # First two samples stays the same
        U1 = np.array(X[:2])
        # Second two we multiply with -1
        U2 = np.multiply(X[2:], -1.0)
        # Combine extended samples in one matrix
        X = np.concatenate((U1, U2),)
        
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

            
            # Calculate the dot product with the sample
            d_12 = np.dot(self.w, sample)
            
            
            # Check which side of the dividing line the sample is located on
            if d_12 > 0:
                y_hat.append(1)
            elif d_12 < 0:
                y_hat.append(2)
            elif d_12 == 0:
                # The sample cannot be sorted
                y_hat.append(0)
    
        return y_hat
    
    def score(self, y_test, y_hat):
        """ The method takes the vector of correct and the vector of recognized samples and returns the proportion of correctly recognized samples. """
        return np.mean(np.array(y_test) == np.array(y_hat))

    def plot_boundary(self, X, y):
        """ The method takes a matrix of samples and a vector of their labels and draws a graph with patterns and a separating boundary. """
        a = self.w[0]
        b = self.w[1]
        c = self.w[2]

        
        # Calculation of the equation of the line of the dividing boundary and
        # drawing a line
        fig = plt.figure(figsize=(10, 8))
        if b == 0:
            line_x = -c/a
            print(f"Decision boundary: x={line_x}")
            plt.axvline(x=line_x, color='r', linestyle='-')
        elif a == 0:
            line_y = -c/b
            print(f"Decision boundary: y={line_y}")
            plt.axhline(y=line_y, color='r', linestyle='-')
        else:
            line_xy = -a/b
            line_xy_c = -c/b
            print(f"Decision boundary: y={line_xy}*x + {line_xy_c}")
            ax = np.linspace(0, 1.5, 100)
            ay = line_xy*ax + line_xy_c
            plt.plot(ax, ay, color='r', linestyle='-')

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], label=cl, alpha=0.8)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title('Graphical representation of the decision boundary')
        plt.legend(loc='right')
        plt.show()

        return a, b, c


def unpack_data(filename):
    """
    Takes the file name, returns list X and list y.
 
    """
    data = []
    X_list = []
    X = []
    y = []
    try:
        with open("data/"+filename, 'r') as f:
            data = f.readlines()

    except IOError:
        print(f"Error opening {filename}")

    data.pop(0)
    
    # Extract data from txt file
    for i, element in enumerate(data):
        if "U1" in element:
            data.pop(i)

        if "U2" in element:
            X_list.append(data[:i])
            data.pop(i)
            X_list.append(data[i:])

    # Generate vector y from X list
    class_num = 1
    for element in X_list:
        [y.append(class_num) for i in element]
        class_num+=1
    # Create a list of lists
    flat_list_X = [item for sublist in X_list for item in sublist]
    
    # Remove \n characters
    flat_list_X = [item.strip() for item in flat_list_X]

    # Split data on ","
    flat_list_X = [item.split(",") for item in flat_list_X]
    
    # Convert string to int
    for idx, sublist in enumerate(flat_list_X):
        flat_list_X[idx] = [
            float(el) if el.isdigit() else el
            for el in sublist
        ]

    print(flat_list_X)
    X = flat_list_X

    return X, y

def test_data(X_test):
    """ 
    The function accepts as an input a matrix with test samples from classes 1 and 2 and a matrix with test samples from classes 1 and 3.
    As a result, it returns a test matrix with 16 test samples from classes 1, 2, and 3. The output matrix is ​​used in testing
    learned boundaries.
     """
    
    # Combine test samples from all three classes
    X_test = np.insert(X, len(X[0]), 1.0, axis=1)

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

    X, y = unpack_data(filename)

    # Test data
    """ X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [1, 1, 2, 2] """

    X = np.array(X)
    y = np.array(y)

    # Initialize the perceptron object and give it a green number of iterations
    perceptron = Perceptron(num_epochs=n_epochs, lr=l_r)

    # Perform the learning of punctuation boundaries
    perceptron.train(X)
    
    # Print the separation boundary between the sample classes
    print(f"Vector of coefficients of the separation boundary: {perceptron.w}")

    # Draw the separating boundary between the sample classes
    perceptron.plot_boundary(X,y)

    # Take learning samples for test samples
    X_test = test_data(X)
    # Calculate the vector of class labels on the learning samples
    y_hat = perceptron.test(X_test)

    # Calculate the accuracy
    score = perceptron.score(y, y_hat)

    # Results
    print(f"Correct labels: {y}")
    print(f"Predicted labels: {y_hat}")
    print(f"Score:\t{score*100:.2f}%")
    print()
    sys.exit(0)
