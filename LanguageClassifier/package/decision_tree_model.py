from __future__ import division

import math
import random
import sys
from collections import deque
import pickle

import pandas as pd


class dt_node:

    def __init__(self, column_index, label):
        self.column_index = column_index
        self.left = None
        self.right = None
        self.label = label
        self.weight = None


class DT:
    def __init__(self):
        self.visited5 = set()

    def Train_DT(self, filename):

        df = self.feature_selection(filename)
        print("Training Decision Tree")
        level = 1
        max_level = 3
        visited_set = set()
        tree_root = self.create_decision_tree(df, visited_set, level, max_level, [])
        print("Storing Tree in file using pickle")
        file = open('dt_pickle', 'wb')
        pickle.dump(tree_root, file)

    def Train_adaboost(self, filename):

        print("Fitting Training data")
        df = self.feature_selection(filename)
        print("Training adaboost model")
        tree = self.adaboost(None, df, [])
        print("Storing Tree in file using pickle")
        file = open('ada_boost_pickle', 'wb')
        pickle.dump(tree, file)
        print("Storage of Tree complete")

    def predict_adaboost(self, filename, features):

        decision_stub_store = pickle.load(filename)
        my_ada_output = []
        for index, row in features.iterrows():
            array = list(row)
            weight_A = 0
            weight_B = 0

            for i in range(len(decision_stub_store)):
                my_answer = (self.predictions(decision_stub_store[i][0], array))
                if my_answer == "A":

                    weight_A += decision_stub_store[i][1]
                else:
                    weight_B += decision_stub_store[i][1]

            if weight_A > weight_B:
                my_ada_output.append("A")
            else:
                my_ada_output.append("B")

        return my_ada_output

    def fit_test_data(self, test_list):
        dutch_words = {"de", "het", "een", "van", "en","dat"}
        english_words = {"and", "the", "an", "of", "it", "was", 'all', 'and', 'did', 'while'}
        unique_dutch = {'ä', 'ö', 'ü'}

        dutch_numbers = {'een', "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"}
        #f = open(filex, encoding="utf8")
        feature_dataframe = pd.DataFrame()
        df5 = pd.DataFrame()
        for x in test_list:
            min_size_one = False  # Feature 1

            Average_word_length_five = False  # Feature 2
            average_count = 0

            contains_ij = False  # Feature 3

            contains_q = False  # Feature 4

            contains_dutch_count = 0
            contains_english_count = 0

            contains_dutch = False  # feature 5
            contains_english_4 = False  # feature 9

            contains_12 = False  # feature 6

            contains_2_z = False  # feature 7
            contains_z_count = 0

            contains_unique_dutch = False  # feature 8

            contains_dutch_number = False  # feature 10

            q_count = 0

            row = x.split(" ")

            target = 0

            arti = False
            feature_sentence = row
            No_of_words = len(feature_sentence)
            sum_of_lengths = 0

            for word in feature_sentence:

                if word in dutch_numbers:
                    contains_dutch_number = True

                if len(word) > 1:
                    if word[0] == 'z':
                        contains_z_count += 1

                if (len(word) >= 11):
                    contains_12 = True

                if word in dutch_words:
                    contains_dutch_count += 1

                if word in english_words:
                    contains_english_count += 1

                if word.isalpha():
                    for i in range(len(word) - 1):

                        if (word[i] in unique_dutch):
                            contains_unique_dutch = True

                        if word[i] == 'i' and word[i + 1] == 'j':
                            contains_ij = True

                        if word[i] == 'Q' or word[i] == 'q':
                            q_count += 1

                            if q_count > 1:
                                contains_q = True

                if word.isalpha() and len(word) == 1:
                    min_size_one = True

                sum_of_lengths = sum_of_lengths + len(word)

            average_count = sum_of_lengths / No_of_words

            # check avg count

            if average_count > 5:
                # print("avg",target)
                Average_word_length_five = True

            # check dutch count
            if contains_dutch_count > 2:
                contains_dutch = True

            if contains_english_count > 1 and contains_dutch_count > 1:
                if contains_english_count > contains_dutch_count:
                    contains_dutch = False
                else:
                    contains_dutch = True

            if contains_english_count >= 4:
                contains_english_4 = True

            if contains_z_count >= 2:
                contains_2_z = True

            x = {"C1": min_size_one, "C2": Average_word_length_five, "C3": contains_ij, "C4": contains_q,
                 "C5": contains_dutch, "C6": contains_12, "C7": contains_2_z, "C8": contains_unique_dutch,
                 "C9": contains_english_4}

            df5 = df5.append(x, ignore_index=True)

        return df5

    def feature_selection(self, examples):

        dutch_words = {"de", "het", "een", "van", "en","dat"}
        english_words = {"and", "the", "an", "of", "it", "was", 'all', 'and', 'did', 'while'}
        unique_dutch = {'ä', 'ö', 'ü'}

        dutch_numbers = {'een', "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"}
        f = open(examples, encoding="utf8")
        feature_dataframe = pd.DataFrame()
        df5 = pd.DataFrame()

        for x in f:
            min_size_one = False  # Feature 1

            Average_word_length_five = False  # Feature 2
            average_count = 0

            contains_ij = False  # Feature 3

            contains_q = False  # Feature 4

            contains_dutch_count = 0
            contains_english_count = 0

            contains_dutch = False  # feature 5
            contains_english_4 = False  # feature 9

            contains_12 = False  # feature 6

            contains_2_z = False  # feature 7
            contains_z_count = 0

            contains_unique_dutch = False  # feature 8

            contains_dutch_number = False  # feature 10

            q_count = 0

            row = x.split('|')

            target = row[0].strip()

            arti = False
            feature_sentence = row[1].split(" ")
            No_of_words = len(feature_sentence)
            sum_of_lengths = 0

            for word in feature_sentence:

                if word in dutch_numbers:
                    contains_dutch_number = True

                if len(word) > 1:
                    if word[0] == 'z':
                        contains_z_count += 1

                if (len(word) >= 11):
                    contains_12 = True

                if word in dutch_words:
                    contains_dutch_count += 1

                if word in english_words:
                    contains_english_count += 1

                if word.isalpha():
                    for i in range(len(word) - 1):

                        if (word[i] in unique_dutch):
                            contains_unique_dutch = True

                        if word[i] == 'i' and word[i + 1] == 'j':
                            contains_ij = True

                        if word[i] == 'Q' or word[i] == 'q':
                            q_count += 1

                            if q_count > 1:
                                contains_q = True

                if word.isalpha() and len(word) == 1:
                    min_size_one = True
                    # print("avg", x)

                # if (word in dutch_articles):
                #    arti = True

                sum_of_lengths = sum_of_lengths + len(word)

            average_count = sum_of_lengths / No_of_words

            # check avg count

            if average_count > 5:
                # print("avg",target)
                Average_word_length_five = True

            # check dutch count
            if contains_dutch_count > 2:
                contains_dutch = True

            if contains_english_count > 1 and contains_dutch_count > 1:
                if contains_english_count > contains_dutch_count:
                    contains_dutch = False
                else:
                    contains_dutch = True

            if contains_english_count >= 4:
                contains_english_4 = True

            if contains_z_count >= 2:
                contains_2_z = True

            if target.strip() == "en":
                target = "A"
            else:
                target = "B"

            x = {"C1": min_size_one, "C2": Average_word_length_five, "C3": contains_ij, "C4": contains_q,
                 "C5": contains_dutch, "C6": contains_12, "C7": contains_2_z, "C8": contains_unique_dutch,
                 "C9": contains_english_4, "target": target}

            df5 = df5.append(x, ignore_index=True)

        return df5

    def label_gen(self, df):
        P_class = 0
        N_class = 0
        prob_of_A = 0
        prob_of_B = 0
        class_entropy = 0
        target_class = list(df)
        for i in range(len(target_class)):

            if target_class[i] == 'A':
                P_class += 1

            elif target_class[i] == 'B':
                N_class += 1

            if N_class > P_class:
                label = "B"
            else:
                label = "A"

        return label

    visited = set()
    level = 1
    max_level = 3

    def predict(self, X, tree):
        output = []

        for index, row in X.iterrows():
            x = list(row)

            output.append(self.predictions(tree, x))
        return output

    def predictions(self, tree, input):
        if not tree.left and not tree.right:
            # print(tree.label)
            return tree.label

        if input[int(tree.column_index[1]) - 1] == True:

            x = self.predictions(tree.left, input)
            return x

        else:
            x = self.predictions(tree.right, input)
            return x

    def calc_entropy(self, df, current_column, target, weight):
        # Calculate class entropy
        # print(("IN ENTRO", len(df)))
        target_class = list(df.iloc[:, -1])

        P_class = 0
        N_class = 0
        class_entropy = 0
        target = list(target)
        for i in range(len(target)):

            if target_class[i] == 'A':
                P_class += 1
                # P_class += weight[i]

            elif target_class[i] == 'B':
                N_class += 1
                # N_class += weight[i]

                #     class_entropy = (-P_class / (P_class + N_class) * math.log(P_class / (P_class + N_class), 2)) - (
        #                N_class / (P_class + N_class) * math.log(N_class / (P_class + N_class), 2))

        # Calculate column_entropy
        PT = 0
        PF = 0
        NT = 0
        NF = 0
        prob_of_A = 0
        prob_of_B = 0

        array = list(current_column)

        for row in range(len(array)):

            if array[row] == True and target[row] == 'A':

                PT += 1
            #   PT += weight[row]

            elif array[row] == False and target[row] == 'A':
                # print("hello2")
                PF += 1
                # PF += weight[row]

            elif array[row] == True and target[row] == 'B':

                # print("hello3")
                NT += 1
                # NT += weight[row]

            elif array[row] == False and target[row] == 'B':
                # print("hello4")
                NF += 1
                # NF += weight[row]

        if PT != 0 and NT != 0:
            True_entropy = (-PT / (PT + NT) * math.log(PT / (PT + NT), 2)) - (
                    NT / (NT + PT) * math.log(NT / (PT + NT), 2))

        else:
            True_entropy = 0

        if PF != 0 and NF != 0:
            False_entropy = (-PF / (PF + NF) * math.log(PF / (PF + NF), 2)) - (
                    NF / (NF + PF) * math.log(NF / (PF + NF), 2))
        else:
            False_entropy = 0

        """ Find entropy """
        column_entropy = ((PT + NT) / (P_class + N_class) * True_entropy) + (
                (PF + NF) / (P_class + N_class) * False_entropy)

        return column_entropy

    def create_decision_tree(self, df, visited_set, level, max_level, wt):
        '''

        This creates decision tree.
        :return: root of decision tree

        '''
        P_class = 0
        N_class = 0
        prob_of_A = 0
        prob_of_B = 0

        if df is None:
            return

        target_class = list(df.iloc[:, -1])

        for i in range(len(target_class)):

            if target_class[i] == 'A':
                P_class += 1
            # P_class += wt[i]
            elif target_class[i] == 'B':
                N_class += 1

            # N_class += wt[i]

        if N_class != 0 and P_class != 0:
            prob_of_A = P_class / (N_class + P_class)
            prob_of_B = N_class / (N_class + P_class)

        if N_class == 0 and P_class != 0:  # indicates that we have only A's as count of B's is zero
            return dt_node("A", 'A')

        elif P_class == 0 and N_class != 0:  # indicates that we have only B's as count of A's is zero
            return dt_node("B", 'B')

        if prob_of_A > 0.95:
            return dt_node("A", 'A')

        elif prob_of_B > 0.95:
            return dt_node('B', 'B')

        # Get all columns
        temp_matrix = df.iloc[:, :-1]

        # Get last column
        target = df.iloc[:, -1]
        # print(df)

        min_entropy = float('inf')

        all_entropy = []
        for col in temp_matrix:
            if col not in self.visited5:
                all_entropy.append(self.calc_entropy(df, temp_matrix[col], target, wt))

            if col in self.visited5:
                all_entropy.append(1)

                # if entropy < min_entropy:
                #   min_entropy = entropy
                #  min_index = col
                # column = df[col]

        for i in range(len(all_entropy)):
            if all_entropy[i] < min_entropy:
                min_entropy = all_entropy[i]
                min_index = df.columns[i]

        label = self.label_gen(target)

        root = dt_node(min_index, label)
        self.visited5.add(min_index)

        if level >= max_level:
            return root

        level += 1

        if min_entropy == 0:
            return root
        split_on = list(set(df[min_index]))
        split_true = None
        split_false = None

        if len(split_on) >= 2:
            split_true = df.loc[df[min_index] == True]

            split_false = df.loc[df[min_index] == False]

        root.left = self.create_decision_tree(split_true, self.visited5, level, max_level, wt)
        root.right = self.create_decision_tree(split_false, self.visited5, level, max_level, wt)

        return root

    def adaboost(self, tree, df, weight_array):
        N = len(df)
        z = []
        z1 = []
        # This array will store initial weights
        for i in range(len(df)):
            weight_array.append(1 / len(df))

        # This array will store the output from our tree
        my_output = []
        N = len(df)
        actual_ans = list(df.iloc[:, -1])
        decision_stub_store = []
        tree3 = None
        dataframe_store = df
        df2 = df

        # max_level=3
        # level=1
        errorx = []
        visited_set = set()
        level = 1
        max_level = 2
        for i in range(9):
            dataframe_store = df2
            temp_matrix_ada = dataframe_store
            pred_matrix = temp_matrix_ada.iloc[:, :-1]

            # actual_ans = list(dataframe_store.iloc[:, -1])
            tree3 = self.create_decision_tree(temp_matrix_ada, visited_set, level, max_level, weight_array)
            my_output = (self.predict(pred_matrix, tree3))

            error = 0.0

            # print(weight_array)
            # print("SUM", math.fsum(weight_array))
            for i in range(N):

                if actual_ans[i] != my_output[i]:
                    error = error + weight_array[i]

            errorx.append(error)
            # print("ERROR", error)
            store = (1 - error) / error
            performance = 0.5 * (math.log(store))

            # print("Mine__", my_output)
            # print("Real__", actual_ans)
            # print("SIG", store)

            # if (error == 0):
            #     decision_stub_store.append((tree3, 100000))
            #    break

            for i in range(N):
                if my_output[i] == actual_ans[i]:
                    # if(weight_array[i]>0):
                    weight_array[i] = weight_array[i] * (math.e ** (-performance))
                else:
                    # if (weight_array[i] > 0):
                    weight_array[i] = weight_array[i] * (math.e ** performance)

            for i in range(len(weight_array)):
                # weight_array[i] = ((weight_array[i] -min(weight_array)) / (max(weight_array)-min(weight_array)))
                weight_array[i] = weight_array[i] / math.fsum(weight_array)

            decision_stub_store.append((tree3, performance))

            # Select records in a random range
            while len(df2) < len(df):

                rangex = random.uniform(0, 1)

                for j in range(len(weight_array) - 1):
                    # print(range_lower , float(weight_array[j]) ,range_upper)
                    if weight_array[j] < weight_array[j + 1]:
                        w_lower = weight_array[j]
                        w_upper = weight_array[j + 1]

                    else:
                        w_lower = weight_array[j + 1]
                        w_upper = weight_array[j]

                    if w_lower < rangex < w_upper:
                        x = dataframe_store.iloc[[j]]
                        #  print(x)
                        df2 = df2.append(x)
                    if len(df2) >= len(df):
                        break

            # print("DF2", len(df2))

        return decision_stub_store
