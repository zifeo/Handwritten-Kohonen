
"""Python script for Exercise set 6 of the Unsupervised and
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plt

def apply_kohonen(data, size_k=6, sigma=3.0, eta=0.005, tmax=5000, decay=False, decay_rate=0.8):
    """Applies a kohonen map on the data with some parameters.
         data      (vector) the data on which to apply the the kohonen map
         size_k    (scalar) the square root of the size of the kohonen map
         sigma     (scalar) width/variance of neighborhood function
         eta       (scalar) learning rate
         tmax      (scalar) max number of iterations
         
         return    (vector) the centers found for the kohonen map
    """  
    # total dimension
    dim = 28 * 28
    # dimension support
    data_range = 255
    
    plt.rcParams['figure.figsize'] = (size_k, size_k)
    
    # centers randomly initialized
    centers = np.random.rand(size_k ** 2, dim) * data_range

    # neighborhood matrix
    neighbor = np.arange(size_k ** 2).reshape((size_k, size_k))

    # random order in which the datapoints should be presented
    i_random = np.arange(tmax) % len(data)
    np.random.shuffle(i_random)

    scores = []
    history = []
    
    if decay:
        sigma_f = lambda x: sigma * np.exp(-1 * x / (decay_rate * tmax))
    else:
        sigma_f = lambda x: sigma

    for t, i in enumerate(i_random):
        # at each iteration, compute the step and store the state
        score = som_step(centers, data[i, :], neighbor, eta, sigma_f(t))
        scores.append(score)

    # show scores
    plt.title('Scores per iteration')
    plt.plot(scores)
    plt.ylabel("score")
    plt.xlabel("iteration")
    plt.axvline(np.argmin(scores), color='red')
    plt.show()

    # visualize prototypes
    plt.title('prototypes at best score')
    for i in range(size_k ** 2):
        plt.subplot(size_k, size_k, i + 1)
        plt.imshow(centers[i,:].reshape([28, 28]), interpolation='bilinear', cmap='Greys')
        plt.axis('off')

    plt.show()
    return centers

def label_assignements(data, labels, centers, size_k, plot_prob=False):
    proto_assign = {}
    closest_proto = []
    
    # for each entry, get closest proto
    for e in data:
        diff = [np.sum(np.square(centers[i, :] - e)) for i in range(size_k ** 2)]
        coord = np.argmin(diff)
        closest_proto.append(coord)
        
    closest_proto = np.array(closest_proto)
    
    for p in range(size_k**2):
        labels_present, counts = np.unique(labels[closest_proto == p], return_counts=True)
        if len(counts) > 0:
            proto_assign[p] = (labels_present, counts, labels_present[np.argmax(counts)])
        else:
            proto_assign[p] = (labels_present, counts, "None")
        
    plt.title('prototypes at best score, with labels')

    for i in range(size_k ** 2):
        plt.subplot(size_k, size_k, i + 1)

        plt.title(proto_assign[i][2])
        plt.imshow(centers[i,:].reshape([28, 28]), interpolation='bilinear', cmap='Greys')
        plt.axis('off')

    plt.show()
    
    if plot_prob:
        plt.title('prototypes at best score, with label confidence (%)')

        for i in range(size_k ** 2):
            plt.subplot(size_k, size_k, i + 1)

            labels_present = proto_assign[i][0]
            counts = proto_assign[i][1]
            tot_counts = np.sum(counts)
            res = ""

            for l,c in zip(labels_present, counts):
                res += str(l) + "("
                res += str(int(c / tot_counts * 100))
                res += ") "

            plt.title(res, fontsize=5)
            plt.imshow(centers[i,:].reshape([28, 28]), interpolation='bilinear', cmap='Greys')
            plt.axis('off')

        plt.show()

def som_step(centers, datapoint, neighbor, eta, sigma):
    """Learning step self-organized map updating inplace centers.
         centers   (matrix) cluster centres (center X dimension)
         datapoint (vector)
         neighbor  (matrix) coordinates of all centers
         eta       (scalar) learning rate
         sigma     (scalar) width/variance of neighborhood function
    """    
    k = np.argmin(np.sum(np.square(centers - datapoint), axis=1))
    k_coords = np.array(np.nonzero(neighbor == k))
        
    for j in range(len(centers)):
        j_coords = np.array(np.nonzero(neighbor == j))
        disc = neighborhood(np.linalg.norm(k_coords - j_coords), 0, sigma)
        centers[j, :] += disc * eta * (datapoint - centers[j, :])
    
    return np.sqrt(np.sum(np.square(centers[k, :] - datapoint))) / len(centers[k,:])
        
def neighborhood(x, mean, std):
    """Normalized neighborhood gaussian-like with mean and std."""
    return np.exp(- np.square(x - mean) / (2 * np.square(std)))

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = int(np.mod(s,x.shape[0]))

    return np.sort(x[t,:])


if __name__ == "__main__":
    kohonen()

