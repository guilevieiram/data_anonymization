import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict

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

    cdef int[:] get_one_hot(self, word: str):
        cdef int[:] x = np.zeros((len(self.vocabulary)))
        x[self.get_vocabulary_index(word)] = 1
        return x

    cdef void fit_knn(self):
        assert("train" in set(self.data.keys()))
        cdef dict word_map = dict()
        for word, ans in self.data["train"]:
            if word_map.get(word) is None: word_map[word] = []
            word_map[word].append(ans)
        self.models["knn"] = word_map

    cdef int predict_knn(self, word: str):
        cdef dict word_map = self.models["knn"]
        if word not in word_map.keys(): return 0
        cdef float mean = sum(word_map[word])/len(word_map[word])
        return int(mean >= 0.5)

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

    def get_X(self): return self.X
    def get_y(self): return self.y

    cpdef void fit(self, model: str):
        assert(model in ["knn"])
        print(f"Fitting {model}")

        if model == "knn":
            self.fit_knn()

    cpdef void predict(self, model: str):
        assert(model in ["knn"])
        assert(self.models[model] is not None)

        cdef list anss = []
        cdef list pred = []
        cdef float acc = 0

        if model == "knn":
            for test in ["testa", "testb"]:
                for word, ans in self.data[test]:
                    anss.append(ans)
                    pred.append(self.predict_knn(word))
                acc = accuracy_score(anss, pred)
                print(f"KNN accuracy in {test}: {(100*acc):.2f}%")
                anss, pred = [], []
