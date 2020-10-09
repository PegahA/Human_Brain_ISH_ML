from human_ISH_config import *
import os

if __name__ == "__main__":

    top_list = [[20, 8e-5, True],
                [17, 7e-5, True],
                [18, 7e-5, True],
                [12, 8e-5, True]]

    training_iterations_list = [30000] #[10000, 15000, 20000]

    main_py_path = os.path.join(CODE_DIR, "main.py")

    for iter in training_iterations_list:
        decay_start = iter - 5000

        for item in top_list:
            print("Here in the top level loop ...")


            batch_p = item[0]
            lr = item[1]
            flip = item[2]

            train_iterations = iter
            decay_start_iteration = decay_start
            batch_k = 300 // batch_p

            main_command_line_string = "python " + main_py_path + \
                                       " --train_batch_p=" + str(batch_p) + \
                                       " --train_batch_k=" + str(batch_k) + \
                                       " --learning_rate=" + str(lr) + \
                                       " --train_iterations=" + str(train_iterations) + \
                                       " --decay_start_iteration=" + str(decay_start_iteration) + \
                                       (" --train_flip_augment=1" if flip else " --train_flip_augment=''")

            os.system(main_command_line_string)

    print("Finished the top level loop ...")


