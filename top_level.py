from human_ISH_config import *
import os



if __name__ == "__main__":


 
    batch_p_list = [17] #[5, 10, 20, 40, 50, 60]

    learning_rate_list = [7e-5, 10e-5]

    flip_list = [True, False]

    distance_list = ['euclidean', 'sqeuclidean', 'cityblock']

    margin_list = ['0.5', '1.0']


    main_py_path = os.path.join(CODE_DIR, "main.py")

    for dist in distance_list:
        for margin in margin_list:
            for lr in learning_rate_list:
                for batch_p in batch_p_list:
                    for flip in flip_list:

                        print ("Here in the top level loop ...")
                        batch_k = 300 // batch_p

                        main_command_line_string = "python " + main_py_path + \
                                            " --train_batch_p=" + str(batch_p) + \
                                            " --train_batch_k=" + str(batch_k) +\
                                            " --learning_rate=" +  str(lr) +\
                                            " --margin=" + margin +\
                                            " --metric=" + dist +\
                                            (" --train_flip_augment=1" if flip else " --train_flip_augment=''")


                        os.system(main_command_line_string)



    print ("Finished the top level loop ...")


