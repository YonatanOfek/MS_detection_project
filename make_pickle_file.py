import pickle


def make_pickle(dict, filename):
    """
    make pickle file
    :param dict: the dictionary
    :param filename: 'trained_dict'
    """
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)





