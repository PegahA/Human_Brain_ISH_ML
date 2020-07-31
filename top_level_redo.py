from human_ISH_config import *
import os

if __name__ == "__main__":


    top_list = [[18, 10e-5, False],
                [18, 7e-5, True],
                [16, 7e-5, True],
                [20, 7e-5, False],
                [20, 8e-5, True],
                [16, 5e-5, False],
                [14, 6e-5, False],
                [20, 9e-5, False],
                [12, 8e-5, True],
                [16, 6e-5, True]]

    main_py_path = os.path.join(CODE_DIR, "main.py")

    for i in range(2):  # 2 times
        for item in top_list:

            batch_p = item[0]
            lr = item[1]
            flip = item[2]

            print("Here in the top level loop ...")
            batch_k = 300 // batch_p

            main_command_line_string = "python " + main_py_path + \
                                       " --train_batch_p=" + str(batch_p) + \
                                       " --train_batch_k=" + str(batch_k) + \
                                       " --learning_rate=" + str(lr) + \
                                       (" --train_flip_augment=1" if flip else " --train_flip_augment=''")

            os.system(main_command_line_string)

    print("Finished the top level loop ...")


