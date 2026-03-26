from utils.utils import *
from utils import utils
import os

"""
Script for phase-shifting and amplitude, phase retrieval 
Controls:
    1) a second screen which displays the sinusoidal pattern, 
    2) a vimba camera from allied vision for acquire the frames 
Calculate:
    1) Phase map
    2) modulated map
Save: in path_to_repo\surface_results\surface
    1) Save the shifted-pattern frames Phase0,Phase1,Phase2,Phase3...
    2) Save the modulated map and thw erapped phase
"""




#---- global variables
initial_freq = 60  # cycles/m                 # initial freq for projected fringes
initial_exposure = 10000  # 10 ms
base_path = os.path.dirname(__file__) # Directory for saving phase-shifting images
path_folder = os.path.join(base_path, f"surface_results/surface") # path 

# creation of the folder for saving images
create_folder(path_folder)

# evaluation of the second screen properties for displaying sinusoidal pattern 
main_monitor_size, window_displacement, resize_window_height, resize_window_width, monitors_size_mm = utils.evaluate_and_detect_monitors()

# the user have to posicionate the camera to observe the fringe projection 
# main loop
while True:
    array_of_images = []

    print("Select the FREQUENCY, ANGLE and camera EXPOSURE TIME to implement the retrieval")
    patterns_to_display, exposure_time = run_camera_and_fringes_ui(freq_ini=initial_freq,
                                                                  second_screen_width=resize_window_width,
                                                                  second_screen_height=resize_window_height,
                                                                  displacement_x=window_displacement,
                                                                  displacement_y=0,
                                                                  monitors_size=monitors_size_mm,
                                                                  main_monitor_size = main_monitor_size,
                                                                  cam_exposure = initial_exposure)

    #  try for testing the camera conection
    run_camera(exposure_time, pixel_format = 'Mono8')
    cam = utils.cam                      # variable for contolling camera features
    vmb = utils.vmb                     # vimba system

    # shifted_frames is a LIST of frames acquired from the camera, 
    shifted_frames = phase_shifting_loop(cam, vmb, patterns_to_display)

    # post-proccesing
    # obatain significant ...phase, phasor, compensated_phase, modulated intensity map... from the acquired frames
    complex_matrix = phasor(shifted_frames)
    amplitude_from_complex = amplitude_from_phasor(complex_matrix)
    phase = phase_calculation_from_array(shifted_frames)

    [array_of_images.append(im) for im in shifted_frames]
    array_of_images.append(amplitude_from_complex)
    array_of_images.append(phase)

    plot_arrays(array_of_images[0],array_of_images[-1], array_of_images[-2])

    save_images = (input("save_images (yes) (no): ")).lower() # ask in terminal to save or not the images
    if save_images in {"si", "sí", "s", "yes", "y", "Y", "YES"}:
        save_8_bit_images(array_of_images, path_folder)  
        close_camera()
        break

    run_again = (input("Run again the retrieval? (yes) (no): ")).lower() # ask in terminal to save or not the images
    if run_again in {"si", "sí", "s", "yes", "y", "Y", "YES"}:
        print("running again")
    else:
        close_camera()
        break


    