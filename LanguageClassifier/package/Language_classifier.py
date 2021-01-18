import pickle
import LanguageClassifier.package.decision_tree_model as decision_tree_model


def test_dt():
    DT = decision_tree_model.DT()

    df = DT.fit_test_data(test_list)
    pickle_file_DT = open('dt_pickle', 'rb')
    root_Node = pickle.load(pickle_file_DT)
    my_answer = (DT.predict(df, root_Node))
    return my_answer

def test_adaboost():
    DT = decision_tree_model.DT()
    df = DT.fit_test_data(test_list)
    pickle_file_ada = open('ada_boost_pickle', 'rb')
    my_answer = DT.predict_adaboost(pickle_file_ada, df)
    return my_answer

def train_both():
    DT = decision_tree_model.DT()
    filename = 'train-corpus'
    DT.Train(filename)
    filename_ada = 'train-corpus'
    DT.Train_adaboost(filename_ada)

test_list = ['linkin park brak in 2000 door met het debuutalbum hybrid theory dat de diamanten status kreeg']

my_answer = test_dt()

for x in my_answer:
    if x == 'A':
        print("en")
    else:
        print("nl")
