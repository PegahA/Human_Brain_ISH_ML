from human_ISH_config import *
import os



if __name__ == "__main__":

    batch_p_list = [5, 10, 20, 40, 50, 60]

    learning_rate_list = [5e-5, 10e-4, 10e-5, 10e-6]

    #flip_list = [TRUE, FALSE]

    main_py_path = os.path.join(CODE_DIR, "Human_Brain_ISH_ML", "main.py")
    for lr in learning_rate_list:
        for batch_p in batch_p_list:

            print ("Here in the top level loop ...")
            batch_k = 300 // batch_p

            main_command_line_string = "python " + main_py_path + \
                                        " --train_batch_p=" + str(batch_p) + \
                                        " --train_batch_k=" + str(batch_k) +\
                                        " --learning_rate=" +  str(lr)
            os.system(main_command_line_string)



    print ("Finished the top level loop ...")


