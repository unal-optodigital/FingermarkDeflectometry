import matplotlib.pyplot as plt
import cv2
import numpy
from functions import *
from PIL import Image
from functions import *
import os
from skimage.restoration import unwrap_phase
from pathlib import Path


# luego de tener esta carpeta de interés leer las imágenes adquiridas en el
# phase shifting, con base en esto, a cada una de las imágenes provenientes del
# phase shifting se le aplica el recorte en el ROI
# Para estas imágenes recortadas, se obtiene la fase y la amplitud
# con la amplitud la idea es meterla directamente en la función de evaluación de contraste
# con la fase la idea es desenvolverla y compensarla lo máximo posible para poder obtener una medida de la visibilidad
surface_type = "Coffecup" # Aluminium_surface, Coffecup, Sink, Stainless steel table top
global_script_path = os.path.dirname(__file__)
folder_path = "/Results/" + surface_type + f"/images/"
path = Path(global_script_path + folder_path)
n_experiments = sum(p.is_dir() for p in path.iterdir())
fingerprints_amplitude_contrast = []
fingerprints_phase_contrast = []
# "'C:\Program Files\Allied Vision\Vimba X\api\pycthon\vmbpy-1.2.0-py3-none-win_amd64.whl

## iteration for the folder of the experiment 1, 2 ,3 ,4 ...
# each folder contains images that which also contains multplefingermarks that must be analized
# for n in range(1,n_experiments+1):
for n in range(7,9):
    #------------------------ entra a la carpeta n, se leen los Phase_i y ampl, se almacena todo en la lista []
    phase_shifting_array = []

    for i in range(0,6): # phase0, ..., phase 4, modulated_map

        if i != 5:
            image_path = global_script_path + folder_path + f"{n}/Phase_{i}.png"
        else:
            image_path = global_script_path + folder_path + f"{n}/modulated_intensity_map.png"
        im = load_image_gray(image_path)
        phase_shifting_array.append(im)  
    
    
    number_of_fingermarks_per_image = imshow_with_textbox_ok(phase_shifting_array[-1], label="radius:", initial="0")
    # number_of_fingermarks_per_image = 1
    # folder for storing Results/surfaceXX/michelson_contrast
    michelson_folder = global_script_path + folder_path + f"{n}/michelson_contrast"
    os.makedirs(michelson_folder, exist_ok=True)

    # counter for saving the selected fingermark 
    fingermark_counter = 1
    # this while is only for the images where are more than 1 fingermark
    while number_of_fingermarks_per_image !=0:
        
        roi_coords = None # Nothing in ROI coords

        roi_coords, roi_array = crop_image(phase_shifting_array[-1]) # shows the amplt for making the CROP in the ROI

        # this are the coords for the crop
        x0, x1, y0, y1 = roi_coords

        # we apply the crop in the same point over the phase-shifted projected patterns
        phase_shifting_array_cropped = [arr[y0:y1, x0:x1] for arr in phase_shifting_array]

        # show the cropped arrays
        # plot_arrays(phase_shifting_array_cropped, titles=["phase0","phase1","phase2","phase3","phase4","Modulated int. map"])
        # Creation of the phasor that contains the information of the 
        complex_matrix = phasor(phase_shifting_array_cropped)

        # amplitude from deflectometry phasor
        amplitude_from_complex = amplitude_from_phasor(complex_matrix)

        # phase from deflectometry phashor
        phase = phase_calculation_from_array(phase_shifting_array_cropped)
        
        # unwrapped phase after substacting the carrier phase
        # unwrapped_phase = normalize_array(unwrap_phase ( 2*np.pi*normalize_array( phase_compensation(phase) ) - np.pi ))
        unwrapped_phase = phase_compensation(phase)


        # this is only for selecting the regions, 3x3 = 9 regions
        nrows, ncols = 3, 3
        list_of_mins_maxs_amplitude, position_clicks, overlay_amplt = segmentation_of_regions(img=amplitude_from_complex, 
                                                                             nrows=nrows, ncols=ncols, 
                                                                             return_overlay = True, stat="mean" )                                          
        list_of_mins_maxs_phase, overlay_phase =  apply_clicks_to_image(img=unwrapped_phase, 
                                                                          nrows=nrows, ncols=ncols, 
                                                                          pos=position_clicks,
                                                                          return_overlay = True, stat="mean") # [ [min,max], [min,max]...]
        


        plot_arrays([overlay_amplt, overlay_phase])

        contrast_amplitude = michelson_contrast_per_region(list_of_mins_maxs_amplitude)
        contrast_phase = michelson_contrast_per_region(list_of_mins_maxs_phase)

        fingerprints_amplitude_contrast.append(mean(contrast_amplitude))
        fingerprints_phase_contrast.append(mean(contrast_phase))
        
            # --------------- GUARDAR INFO MIN MAX DE LAS REGIONES
            # --------------- GUARDAR PROMEDIO DE CONTRASTE EN LA HUELLA


        


        fingermark_folder = michelson_folder + f"/fingermark_{fingermark_counter}"
        print("fingermarkfolderdebugging",fingermark_folder)
        os.makedirs(fingermark_folder, exist_ok=True)
        path_for_amplitude  = fingermark_folder + f"/amplitude_fingermark_{fingermark_counter}.png"
        path_for_phase      = fingermark_folder + f"/phase_fingermark_{fingermark_counter}.png"

        save_minmax_txt(list_of_mins_maxs_amplitude, fingermark_folder + f'/contrast_amplitude{fingermark_counter}.txt')
        save_minmax_txt(list_of_mins_maxs_phase, fingermark_folder + f'/contrast_phase{fingermark_counter}.txt')

        amplitude_from_complex, unwrapped_phase = np.uint8(255*normalize_array(amplitude_from_complex)), np.uint8(255*normalize_array(phase))

        cv2.imwrite(path_for_amplitude,overlay_amplt)
        cv2.imwrite(path_for_phase, overlay_phase)

        number_of_fingermarks_per_image -=1
        fingermark_counter +=1


        # necesito guardar:
        # -datos de min and max per region   AQUÍ: por cada huella habrían rows x cols regiones     
        # estrategico guardar esta vaina en un solo .txt con encabezado de Reg 1, Reg 2, etc... 

        # -promedio por region
        # un txt tambien con Reg 1, Reg 2, con el promedio al lado
        # -imagen phase, imagen amplt con las regiones
        #
        



    # plt.plot(fingerprints_amplitude_contrast);plt.title('phase_contrast');plt.show()
    # plt.plot(fingerprints_phase_contrast);plt.title('amplitude contrast');plt.show()

    # Here is saved the mean of the fingermaks
    save_txt(y_data=fingerprints_amplitude_contrast, file_name= global_script_path + folder_path + f"ampl_contrast.txt")
    save_txt(y_data=fingerprints_phase_contrast,     file_name= global_script_path + folder_path + f"phase_contrast.txt")
 