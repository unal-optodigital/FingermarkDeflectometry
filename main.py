import functions
from functions import *
import os

#---- global variables5
freq = 7                   # initial freq for projected fringes
displacement, resize_h, resize_w, = monitor_info()
base_path = os.path.dirname(__file__) # Directorio para save_images las imágenes
path_folder = os.path.join(base_path, f"Results/surface")
create_folder(path_folder)


# the user have to posicionate the camera to observe the fringe projection 

# main loop
while True:
    array_of_images = []

    patterns_to_project,exposure_time = run_camera_and_fringes_ui(freq,resize_w,resize_h,resize_w,resize_h,displacement)

    #  try for testing the camera conection
    try:
        init_camera(exposure_time, pixel_format = 'Mono8')
        cam = functions.cam                      # variable for contolling camera features
        vmb = functions.vmb                     # vimba system
        
    except:
        print("camera is not connected.")
        close_camera()
        exit()

    shifted_frames = phase_shifting_loop(cam, vmb, patterns_to_project, path_folder)

    # post-proccesing
    # obatain significant ...phase, phasor, compensated_phase, modulated intensity map... from the acquired frames
    complex_matrix = phasor(shifted_frames)
    amplitude_from_complex = amplitude_from_phasor(complex_matrix)
    phase = phase_calculation_from_array(shifted_frames)

    [array_of_images.append(im) for im in shifted_frames]
    array_of_images.append(amplitude_from_complex)
    array_of_images.append(phase)


    #save_images = (input("save_images (yes) (no): ")).lower()
    # save_images
    save_images = 'si'
    if save_images in {"si", "sí", "s", "yes", "y"}:
        save_8_bit_images(array_of_images,path_folder)  


    # continue_the_retrieval = input("Make the retrieval again? (yes) (no): ") 
    continue_the_retrieval = 'yes'
    # evaluate if the program continues or not
    if continue_the_retrieval.lower() != 'yes':
        close_camera()
        break
    