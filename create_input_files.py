from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='data/caption_data_0_100.csv', 
                       image_folder='data/images/', 
                       captions_per_image=1000, 
                       min_word_freq=0, 
                       output_folder='data/proc_data_files/', 
                       max_len=20, 
                       key_max_len=10, 
                       num_images_to_train=10)
