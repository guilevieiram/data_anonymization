import numpy as np
from sklearn.metrics import accuracy_score
from libc.math cimport exp, sqrt, pow

cdef double sigmoid(x: float):
    return 1 / (1 + exp(-x))

cdef double distance(double[:] x, double[:] y):
    assert (len(x) == len(y))
    cdef double dist = 0
    for i in range(len(x)):
        dist += (x[i] - y[i])*(x[i] - y[i])
    return sqrt(dist)

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

    cdef void fit_logistic(self):
        assert("train" in set(self.data.keys()))
        cdef double[:] theta = np.ones(len(self.vocabulary) + 1) ## theta zero
        cdef double[:] new_theta = theta[:] ## theta zero
        cdef double step = 1e-2
        cdef double num_points = len(self.data["train"])
        cdef int max_iters =  2000
        cdef double error = 0
        cdef int coord = 0
        cdef double threshold = 0

        for it in range(max_iters):
            for word, ans in self.data["train"]:
                coord = self.get_vocabulary_index(word) + 1
                if coord < 0: continue
                new_theta[coord] -= step / num_points * (sigmoid(new_theta[coord]) - ans)
                new_theta[0] -= step / num_points * (sigmoid(new_theta[0]) - ans)

            error = distance(theta, new_theta)
            theta = new_theta

            if it % 200 == 0:
                print(f"{(100*it/max_iters):.2f}%")


        for word, ans in self.data["train"]:
            threshold += ans
        threshold /= num_points
        threshold = 1 - threshold

        self.models["logistic"] = (theta, threshold)
        

    cdef int predict_logistic(self, word: str):
        assert("logistic" in set(self.models.keys()))
        cdef int coord = self.get_vocabulary_index(word) + 1
        cdef double pred = 0
        cdef double[:] theta = self.models["logistic"][0]
        cdef double threshold = self.models["logistic"][1]

        pred += sigmoid(theta[coord])
        pred += sigmoid(theta[0])

        return pred >= threshold

    # public methods
    cpdef read_data(self, path: str, data_tag: str):
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


    # main API
    cpdef fit(self, model: str):
        assert(model in ["knn", "logistic"])
        print(f"Fitting {model}")

        if model == "knn":
            self.fit_knn()
        
        if model == "logistic":
            self.fit_logistic()

    cpdef predict(self, model: str):
        assert(model in ["knn", "logistic"])
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

        if model == "logistic":
            for test in ["testa", "testb"]:
                for word, ans in self.data[test]:
                    anss.append(ans)
                    pred.append(self.predict_logistic(word))
                acc = accuracy_score(anss, pred)
                print(f"Logistic accuracy in {test}: {(100*acc):.2f}%")
                anss, pred = [], []