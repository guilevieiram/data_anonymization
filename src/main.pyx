import numpy as np

cdef class Classifier:
    cdef set dictionary
    cdef list vocabulary 
    cdef dict vocabulary_map
    cdef dict data
    cdef dict models

    def __init__(self):
        self.dictionary = set()
        self.data = {}
        self.vocabulary = []
        self.vocabulary_map = {}
        self.models = {}

    cdef int get_vocabulary_index(self, word: str):
        return self.vocabulary_map[word] or -1

    cdef str get_vocabulary_word(self, index: int):
        assert(index >= 0 and index < len(self.vocabulary))
        return self.vocabulary[index]


    # public methods
    cpdef void read_data(self, path: str, data_tag: str):
        cdef list data = []
        self.data[data_tag] = data
        with open(path, "r") as f:
            for line in f:
                line_content = line.split()
                if not line_content: continue
                name, _, _, ner = line_content
                self.dictionary.add(name)
                data.append((
                    name, 
                    1 if ner == "I-PER" else 0
                ))

    cpdef void initialize_vocabulary(self):
        self.vocabulary = list(self.dictionary)
        for i, word in enumerate(self.vocabulary):
            self.vocabulary_map[word] = i

    cpdef set get_dictionary(self):
        return self.dictionary

    cpdef list get_vocabulary(self):
        return self.vocabulary

    cpdef int[:] get_one_hot(self, word: str):
        cdef int[:] x = np.zeros((len(self.vocabulary)))
        x[self.get_vocabulary_index(word)] = 1
        return x

    def get_X(self): return self.X
    def get_y(self): return self.y


    cpdef void fit(self, model: str):
        ...

    cpdef void predict(self, model: str):
        assert(self.models[model] is not None)
        ...