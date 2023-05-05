import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from libc.math cimport exp, sqrt, pow

def evaluate_model(labels, pred_labels):
    try:
        cm = confusion_matrix(labels, pred_labels)
        print(f"Confusion Matrix\n {cm}")

        acc = accuracy_score(labels, pred_labels)
        print(f"Accuracy: \n\t{acc}")

        recall = recall_score(labels, pred_labels)
        print(f"Recall: \n\t{recall}")

        precision = precision_score(labels, pred_labels)
        print(f"Precision: \n\t{precision}")

        f1 = f1_score(labels, pred_labels)
        print(f"F1 Score: \n\t{f1}")

        roc_auc = roc_auc_score(labels, pred_labels)
        print(f"ROC AUC Score: \n\t{roc_auc}")
    except Exception as e:
        print(f"Error calculating metrics: {e}")

cdef double sigmoid(x: float):
    return 1 / (1 + exp(-x))

cdef double distance(double[:] x, double[:] y):
    assert (len(x) == len(y))
    cdef double dist = 0
    for i in range(len(x)):
        dist += (x[i] - y[i])*(x[i] - y[i])
    return sqrt(dist)


cdef class KNN: 
    cdef dict model

    def __init__(self):
        self.model = dict()

    cdef void fit(self, data):
        cdef dict word_map = dict()
        for word, ans in data:
            if word_map.get(word) is None: word_map[word] = []
            word_map[word].append(ans)
        self.model = word_map

    cdef int predict(self, word):
        idx = self.model.get(word)
        if idx is None: return 0
        cdef float mean = sum(idx) / len(idx)
        return int(mean >= 0.5)
    
cdef class Logistic:
    cdef double[:] theta
    cdef double threshold
    cdef dict vocab

    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.theta = np.zeros(len(self.vocab)+ 1) 
        self.threshold = 0

    
    cdef int get_vocabulary_index(self, word: str):
        return self.vocab[word] or -1

    cdef void fit(self, data):
        cdef double[:] theta = np.zeros(len(self.vocab)+ 1) ## theta zero
        cdef double[:] new_theta = theta[:] ## theta zero

        cdef int max_iters = 50
        cdef double error = 0
        cdef int coord = 0
        cdef double threshold = 0
        cdef double min_err = 1e-4
        cdef int count = 0

        cdef double[2][2] hess = np.ones((2, 2))
        cdef double[2] grad = np.ones(2)
        cdef double[2] sol = np.zeros(2)

        cdef double lamb = 1e-3

        for it in range(max_iters):
            count = 0

            for word, ans in data:
                coord = self.get_vocabulary_index(word) + 1
                if coord < 0: continue
                count += 1

                sig = sigmoid(new_theta[0] + new_theta[coord])

                # calculate the hessian
                hess = - np.ones((2, 2)) * sig * (1 - sig) - np.eye(2) * 2 * lamb

                # calculate the grad
                grad = ans - sig - 2 * lamb * np.array([new_theta[0], new_theta[coord]])

                # solve the linear system
                sol = np.linalg.solve(hess, grad)
                error += np.linalg.norm(sol)

                # update theta
                new_theta[0] -= sol[0]
                new_theta[coord] -= sol[1]

                self.theta = new_theta

            error /= count
            print(f"{(100*it/max_iters):.2f}%   error: {error}")
            if error < min_err: break

        for word, ans in data: threshold += ans
        threshold /= len(data)
        self.threshold = threshold

    cdef int predict(self, word):
        cdef int coord = self.get_vocabulary_index(word) + 1
        cdef double pred = sigmoid(self.theta[coord] + self.theta[0])
        return pred >= self.threshold


cdef class Classifier:
    cdef set dictionary
    cdef list vocabulary 
    cdef dict vocabulary_map
    cdef dict data
    cdef KNN knn
    cdef Logistic logistic

    def __init__(self):
        self.dictionary = set()
        self.data = {}
        self.vocabulary = []
        self.vocabulary_map = {}

        self.knn = KNN()
        self.logistic = Logistic(self.vocabulary_map)

    cdef int get_vocabulary_index(self, word: str):
        return self.vocabulary_map[word] or -1

    cdef str get_vocabulary_word(self, index: int):
        assert(index >= 0 and index < len(self.vocabulary))
        return self.vocabulary[index]

    cdef int[:] get_one_hot(self, word: str):
        cdef int[:] x = np.zeros((len(self.vocabulary)))
        x[self.get_vocabulary_index(word)] = 1
        return x

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
        
        self.logistic.vocab = self.vocabulary_map

    cpdef set get_dictionary(self):
        return self.dictionary

    cpdef list get_vocabulary(self):
        return self.vocabulary


    # main API
    cpdef fit(self, model: str):
        assert(model in ["knn", "logistic"])
        print(f"Fitting {model}")

        if model == "knn":
            self.knn.fit(self.data["train"])
        
        if model == "logistic":
            self.logistic.fit(self.data["train"])

    cpdef predict(self, model: str):
        cdef list anss = []
        cdef list pred = []

        if model == "knn":
            for test in ["testa", "testb"]:
                for word, ans in self.data[test]:
                    anss.append(ans)
                    pred.append(self.knn.predict(word))
                print(f"KNN on {test}: ")
                evaluate_model(pred, anss)
                anss, pred = [], []

        if model == "logistic":
            for test in ["testa", "testb"]:
                for word, ans in self.data[test]:
                    anss.append(ans)
                    pred.append(self.logistic.predict(word))
                print(f"Logistic on {test}: ")
                evaluate_model(pred, anss)
                anss, pred = [], []
