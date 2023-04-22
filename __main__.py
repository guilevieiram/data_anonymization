import main

c = main.Classifier()
c.read_data("./dataset/eng.train", "train")
c.read_data("./dataset/eng.testa", "testa")
c.read_data("./dataset/eng.testb", "testb")

c.initialize_vocabulary()

c.initialize_data("train")

print(c.get_X())
print(c.get_y())

# print(c.get_vocabulary_index("Norton"))
# d = c.get_dictionary()

# c1 = main.Classifier()
# c1.read_data("./dataset/eng.testa")
# d1 = c1.get_dictionary()

# c2 = main.Classifier()
# c2.read_data("./dataset/eng.testb")
# d2 = c2.get_dictionary()

# for x in d1: 
#     if x not in d: print("fundeu")

# for x in d2: 
#     if x not in d: print("fundeu")

