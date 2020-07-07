import os

training_case_txt_50 ="/projects/shart/digital_pathology/data/PenMarking/WSIs/training_cases_50.txt"
lines = open(training_case_txt_50, 'r').readlines()

txt_file = os.path.split(training_case_txt_50)
K_fold = 5
for k in range(K_fold):
    all_data = lines.copy()
    testing_lines_start = k*10
    testing_lines_end = (k+1)*10
    print(testing_lines_start)
    print(testing_lines_end)
    print("------------")
    testing_data = all_data[testing_lines_start:testing_lines_end]
    del all_data[testing_lines_start:testing_lines_end]
    training_data = all_data
    train_cv_txt = os.path.join(txt_file[0], "training_cases_50_cv_" + str(k) + ".txt")
    test_cv_txt = os.path.join(txt_file[0], "testing_cases_50_cv_" + str(k) + ".txt")
    with open(train_cv_txt, 'a') as fp:
        for l in training_data:
            fp.write(l)
    with open(test_cv_txt, 'a') as fp:
        for l in testing_data:
            fp.write(l)











