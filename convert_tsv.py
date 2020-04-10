# # with open("datasets/pku_test_gold.utf8") as f, open("datasets/test.tsv", "w") as f_train:
# with open("datasets/pku_training.utf8") as f, open("datasets/train.tsv", "w") as f_train:
#     lines = f.readlines()
#     # lines = lines[:1000]
#     for line in lines:
#
#         line = line.strip().split()
#         for word in line:
#             if len(word) == 1:
#                 f_train.write(word + "\t" + "S" + "\n")
#             else:
#                 for i, char in enumerate(word):
#                     if i == 0:
#                         f_train.write(char + "\t" + "B" + "\n")
#                     elif i == len(word) -1:
#                         f_train.write(char + "\t" + "E" + "\n")
#                     else:
#                         f_train.write(char + "\t" + "I" + "\n")
#         f_train.write("\n")
#
#
#
#     # print(line)

with open("models/checkpoint/ngram.txt") as f, open("models/checkpoint/ngram2.txt","w") as f_train:
    lines = f.readlines()
    for line in lines:
        if not line.strip():continue
        f_train.write(line)