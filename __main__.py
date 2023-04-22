import main

c = main.Classifier()
c.read_data("./dataset/eng.train", "train")
c.read_data("./dataset/eng.testa", "testa")
c.read_data("./dataset/eng.testb", "testb")

c.initialize_vocabulary()

c.fit("knn")
c.predict("knn")

c.fit("logistic")
c.predict("logistic")

