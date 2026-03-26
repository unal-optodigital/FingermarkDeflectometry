import cv2
from utils.utils import *
import os
from pathlib import Path
import utils.ampl_phase_plot as ampl_phase_plot
from utils.csv_save_data import collect_fingermark_summaries, save_outputs


surface_type = "Select_the_name_of_the_surface_folder" # "Coffecup", "Sink"
surface_type = "Sink" # "Coffecup", "Sink"
ampl_phase_plot.surface_type = surface_type
global_script_path = os.path.dirname(__file__)
images_path = "/surface_results/" + surface_type + f"/images/"
path = Path(global_script_path + images_path)
n_experiments = sum(p.is_dir() for p in path.iterdir())
fingerprints_amplitude_contrast = []
fingerprints_phase_contrast = []
save_info = False
local_correction = False
analysis_over_different_donors = False



## iteration for the folder of the experiment 1, 2 ,3 ,4 ...
# each folder contains imaqges that which also contains multplefingermarks that must be analized
if analysis_over_different_donors:
    if local_correction:

        """SE DEBE SELECCIONAR LA IMAGEN DONDE ESTÁ LA HUELLA"""
        capture_correction = 1    # seleccion de imagen donde se encuentra la huella Carpeta1/ huellas    Carpeta2/2huellas  Carpeta3/1huellas
        fingermark = 8
        # seleccion de huella

                                #  en especifico (CORRESPONDE A UN DESFASE EN LINEAS DEL 
                                # .TXT, LA HUELLA 1 ESTA EN LA LINEA 2)
        
        print("Correction on: " + surface_type)
        print("Image that contains the fingermark: ",capture_correction )
        print("Fingermark: ", fingermark)
        print("phase and ampl .txt: ", fingermark + 1)
        

        phase_shifting_array = []

        for i in range(0,6): # phase0, ..., phase 4, modulated_map
            if i != 5:
                image_path = global_script_path + images_path + f"{capture_correction}/Phase_{i}.png"
            else:
                image_path = global_script_path + images_path + f"{capture_correction}/modulated_intensity_map.png"
            im = load_image_gray(image_path)
            phase_shifting_array.append(im)  


        number_of_fingermarks_per_image = 1
        # folder for storing Results/surfaceXX/michelson_contrast
        michelson_folder = global_script_path + images_path + "/michelson_contrast"
        os.makedirs(michelson_folder, exist_ok=True)
        
        # counter for saving the selected fingermark 
    
        roi_coords = None # Nothing in ROI coords
        roi_coords, roi_array = crop_image(phase_shifting_array[-1]) # shows the amplt for making the CROP in the ROI
        # this are the coords for the crop
        x0, x1, y0, y1 = roi_coords
        phase_shifting_array_cropped = [arr[y0:y1, x0:x1] for arr in phase_shifting_array]
        
        complex_matrix = phasor(phase_shifting_array_cropped)
        # amplitude from deflectometry phasor  ddf
        amplitude_from_complex = amplitude_from_phasor(complex_matrix)
        # phase from deflectometry phashor
        phase = phase_calculation_from_array(phase_shifting_array_cropped)
        plot_arrays(phase, amplitude_from_complex)
        unwrapped_phase = phase_compensation(phase)


        # unwrapped_phase = normalize_array(unwrap_phase ( 2*np.pi*normalize_array( phase_compensation(phase) ) - np.pi ))



        nrows, ncols = 3, 3
        list_of_mins_maxs_amplitude, position_clicks, overlay_amplt = segmentation_of_regions(img=amplitude_from_complex, nrows=nrows, 
                                                                        ncols=ncols, 
                                                                        return_overlay = True, 
                                                                        stat="mean" )                                          
        list_of_mins_maxs_phase, overlay_phase =  apply_clicks_to_image(img=unwrapped_phase, nrows=nrows, 
                                                                        ncols=ncols, 
                                                                        pos=position_clicks,
                                                                        return_overlay = True, 
                                                                        stat="mean") # [ [min,max], [min,max]...]

        plot_arrays([overlay_amplt, overlay_phase])


        contrast_amplitude = michelson_contrast_per_region(list_of_mins_maxs_amplitude)
        contrast_phase = michelson_contrast_per_region(list_of_mins_maxs_phase)
        
        fingermark_folder = michelson_folder + f"/fingermark_{fingermark}"
        os.makedirs(fingermark_folder, exist_ok=True)
        path_for_amplitude  = fingermark_folder + f"/amplitude_fingermark_{fingermark}.png"
        path_for_phase      = fingermark_folder + f"/phase_fingermark_{fingermark}.png"
        path_to_ampl_txt = global_script_path + images_path + f"ampl_contrast.txt"
        path_to_phase_txt = global_script_path + images_path + f"phase_contrast.txt"

        if save_info and fingermark >=0:


            amplitude_from_complex, unwrapped_phase = np.uint8(255*normalize_array(amplitude_from_complex)), np.uint8(255*normalize_array(phase))

            cv2.imwrite(path_for_amplitude, overlay_amplt)
            cv2.imwrite(path_for_phase, overlay_phase)

            line_to_be_replaced = fingermark
            mean_amplt = mean(contrast_amplitude)
            mean_phase = mean(contrast_phase)

            # reempalzar lineas en txt de promedios
            reemplazar_linea(path_to_ampl_txt, line_to_be_replaced, f"{fingermark} " + f"{mean_amplt} +")
            reemplazar_linea(path_to_phase_txt, line_to_be_replaced,  f"{fingermark} " + f"{mean_phase} +")

            # guardar datos de min y max por region, 9 regiones actualmente
            save_minmax_txt(list_of_mins_maxs_amplitude, fingermark_folder + f'/contrast_amplitude{fingermark}.txt')
            save_minmax_txt(list_of_mins_maxs_phase, fingermark_folder + f'/contrast_phase{fingermark}.txt')

            #   re-guardar csv
            df_amp, df_phase = collect_fingermark_summaries(michelson_folder)
            amp_csv, phase_csv = save_outputs(michelson_folder, df_amp, df_phase)
            print("\nResumen amplitud:");print(df_amp.head())
            print("\nResumen fase:");print(df_phase.head())

                        # Comprobar si es un archivo
            if os.path.isfile( michelson_folder+"/grading.csv"):
                print(" ")
            else:
                print("the file for unil grading doesnot exist.")
                with open("vacio.csv", "w") as archivo:
                    pass  # No hace nada, solo lo crea
            ampl_phase_plot.make_csv_plot(amp_csv, phase_csv, michelson_folder+"/grading.csv")

    else:

        fingermark_counter = 1 # inicia en la primer huella 
        
        for n in range(1, n_experiments + 1 ):
            #------------------------ entra a la carpeta n, se leen los Phase_i y ampl, se almacena todo en la lista []
            phase_shifting_array = []

            # Leer 
            for i in range(0,6): # phase0, ..., phase 4, modulated_map

                if i != 5:
                    image_path = global_script_path + images_path + f"{n}/Phase_{i}.png"
                else:
                    image_path = global_script_path + images_path + f"{n}/modulated_intensity_map.png"
                im = load_image_gray(image_path)
                phase_shifting_array.append(im)  
            
            
        

            number_of_fingermarks_per_image = imshow_with_textbox_ok(phase_shifting_array[-1], label="radius:", initial="0")

            # folder for storing Results/surfaceXX/michelson_contrast
            michelson_folder = global_script_path + images_path + f"michelson_contrast"
            os.makedirs(michelson_folder, exist_ok=True)

            # counter for saving the selected fingermark 
            
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
                unwrapped_phase = normalize_array(unwrap_phase ( 2*np.pi*normalize_array( phase_compensation(phase) ) - np.pi ))
                # unwrapped_phase = phase_compensation(phase)


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

                if save_info:
                    save_minmax_txt(list_of_mins_maxs_amplitude, fingermark_folder + f'/contrast_amplitude{fingermark_counter}.txt')
                    save_minmax_txt(list_of_mins_maxs_phase, fingermark_folder + f'/contrast_phase{fingermark_counter}.txt')

                amplitude_from_complex, unwrapped_phase = np.uint8(255*normalize_array(amplitude_from_complex)), np.uint8(255*normalize_array(phase))

                if save_info:
                    cv2.imwrite(path_for_amplitude,overlay_amplt)
                    cv2.imwrite(path_for_phase, overlay_phase)

                number_of_fingermarks_per_image -=1
                fingermark_counter +=1

            # Here is saved the mean of the fingermaks
            if save_info:
                save_txt(y_data=fingerprints_amplitude_contrast, file_name= michelson_folder + f"/ampl_contrast.txt")
                save_txt(y_data=fingerprints_phase_contrast,     file_name= michelson_folder + f"/phase_contrast.txt")
                

                #   re-guardar csv
                df_amp, df_phase = collect_fingermark_summaries(michelson_folder)
                amp_csv, phase_csv = save_outputs(michelson_folder, df_amp, df_phase)
                print("\nResumen amplitud:");print(df_amp.head())
                print("\nResumen fase:");print(df_phase.head())\
                
                                # Comprobar si es un archivo
                if os.path.isfile( michelson_folder+"/grading.csv"):
                    print(" ")
                else:
                    print("the file for unil grading doesnot exist.")
                    with open("vacio.csv", "w") as archivo:
                        pass  # No hace nada, solo lo crea
                # hacer plot completo
                try:
                    ampl_phase_plot.make_csv_plot(amp_csv, phase_csv, michelson_folder+"/grading.csv")
                except:
                    pass


else: 
    if local_correction:

        """SE DEBE SELECCIONAR LA IMAGEN DONDE ESTÁ LA HUELLA"""
        capture_correction = 1    # seleccion de imagen donde se encuentra la huella Carpeta1/2huellas    Carpeta2/2huellas  Carpeta3/1huellas
        fingermark = 1
        # seleccion de huella

                                #  en especifico (CORRESPONDE A UN DESFASE EN LINEAS DEL 
                                # .TXT, LA HUELLA 1 ESTA EN LA LINEA 2)
        
        print("CORRECCION LOCAL INICIALIZADA SOBRE LA SUPERFICIE: " + surface_type)
        print("IMAGEN QUE CONTIENE LA HUELLA: ",capture_correction )
        print("Huella a modificar valores de contraste: ", fingermark)
        print("Linea que se modificará en phase y ampl .txt: ", fingermark + 1)
        

        phase_shifting_array = []

        for i in range(0,6): # phase0, ..., phase 4, modulated_map
            if i != 5:
                image_path = global_script_path + images_path + f"{capture_correction}/Phase_{i}.png"
            else:
                image_path = global_script_path + images_path + f"{capture_correction}/modulated_intensity_map.png"
            im = load_image_gray(image_path)
            phase_shifting_array.append(im)  


        number_of_fingermarks_per_image = imshow_with_textbox_ok(phase_shifting_array[-1], label="radius:", initial="0")
        # folder for storing Results/surfaceXX/michelson_contrast
        michelson_folder = global_script_path + images_path + "/michelson_contrast"
        os.makedirs(michelson_folder, exist_ok=True)
        
        # counter for saving the selected fingermark 
    
        roi_coords = None # Nothing in ROI coords
        roi_coords, roi_array = crop_image(phase_shifting_array[-1]) # shows the amplt for making the CROP in the ROI
        # this are the coords for the crop
        x0, x1, y0, y1 = roi_coords
        phase_shifting_array_cropped = [arr[y0:y1, x0:x1] for arr in phase_shifting_array]
        
        complex_matrix = phasor(phase_shifting_array_cropped)
        # amplitude from deflectometry phasor  ddf
        amplitude_from_complex = amplitude_from_phasor(complex_matrix)
        # phase from deflectometry phashor
        phase = phase_calculation_from_array(phase_shifting_array_cropped)
        plot_arrays(phase, amplitude_from_complex)
        unwrapped_phase = phase_compensation(phase)


        # unwrapped_phase = normalize_array(unwrap_phase ( 2*np.pi*normalize_array( phase_compensation(phase) ) - np.pi ))



        nrows, ncols = 3, 3
        list_of_mins_maxs_amplitude, position_clicks, overlay_amplt = segmentation_of_regions(img=amplitude_from_complex, nrows=nrows, 
                                                                        ncols=ncols, 
                                                                        return_overlay = True, 
                                                                        stat="mean" )                                          
        list_of_mins_maxs_phase, overlay_phase =  apply_clicks_to_image(img=unwrapped_phase, nrows=nrows, 
                                                                        ncols=ncols, 
                                                                        pos=position_clicks,
                                                                        return_overlay = True, 
                                                                        stat="mean") # [ [min,max], [min,max]...]

        plot_arrays([overlay_amplt, overlay_phase])


        contrast_amplitude = michelson_contrast_per_region(list_of_mins_maxs_amplitude)
        contrast_phase = michelson_contrast_per_region(list_of_mins_maxs_phase)
        
        fingermark_folder = michelson_folder + f"/fingermark_{fingermark}"
        os.makedirs(fingermark_folder, exist_ok=True)
        path_for_amplitude  = fingermark_folder + f"/amplitude_fingermark_{fingermark}.png"
        path_for_phase      = fingermark_folder + f"/phase_fingermark_{fingermark}.png"
        path_to_ampl_txt = global_script_path + images_path + f"ampl_contrast.txt"
        path_to_phase_txt = global_script_path + images_path + f"phase_contrast.txt"

        if save_info and fingermark >=0:


            amplitude_from_complex, unwrapped_phase = np.uint8(255*normalize_array(amplitude_from_complex)), np.uint8(255*normalize_array(phase))

            cv2.imwrite(path_for_amplitude, overlay_amplt)
            cv2.imwrite(path_for_phase, overlay_phase)

            line_to_be_replaced = fingermark
            mean_amplt = mean(contrast_amplitude)
            mean_phase = mean(contrast_phase)

            # reempalzar lineas en txt de promedios
            reemplazar_linea(path_to_ampl_txt, line_to_be_replaced, f"{fingermark} " + f"{mean_amplt} +")
            reemplazar_linea(path_to_phase_txt, line_to_be_replaced,  f"{fingermark} " + f"{mean_phase} +")

            # guardar datos de min y max por region, 9 regiones actualmente
            save_minmax_txt(list_of_mins_maxs_amplitude, fingermark_folder + f'/contrast_amplitude{fingermark}.txt')
            save_minmax_txt(list_of_mins_maxs_phase, fingermark_folder + f'/contrast_phase{fingermark}.txt')

            #   re-guardar csv
            df_amp, df_phase = collect_fingermark_summaries(michelson_folder)
            amp_csv, phase_csv = save_outputs(michelson_folder, df_amp, df_phase)
            print("\nResumen amplitud:");print(df_amp.head())
            print("\nResumen fase:");print(df_phase.head())


            # Comprobar si es un archivo
            if os.path.isfile( michelson_folder+"/grading.csv"):
                print(" ")
            else:
                print("the file for unil grading doesnot exist.")
                with open("vacio.csv", "w") as archivo:
                    pass  # No hace nada, solo lo crea

            ampl_phase_plot.make_csv_plot(amp_csv, phase_csv, michelson_folder+"/grading.csv")

    else:

        fingermark_counter = 1 # inicia en la primer huella 
        
        for n in range(1, n_experiments + 1 ):
            #------------------------ entra a la carpeta n, se leen los Phase_i y ampl, se almacena todo en la lista []
            phase_shifting_array = []

            # Leer 
            for i in range(0,6): # phase0, ..., phase 4, modulated_map

                if i != 5:
                    image_path = global_script_path + images_path + f"{n}/Phase_{i}.png"
                else:
                    image_path = global_script_path + images_path + f"{n}/modulated_intensity_map.png"
                im = load_image_gray(image_path)
                phase_shifting_array.append(im)  
            
            
        

            number_of_fingermarks_per_image = imshow_with_textbox_ok(phase_shifting_array[-1], label="radius:", initial="0")
            # number_of_fingermarks_per_image = 1
            # folder for storing Results/surfaceXX/michelson_contrast
            michelson_folder = global_script_path + images_path + f"michelson_contrast"
            os.makedirs(michelson_folder, exist_ok=True)

            # counter for saving the selected fingermark 
            
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
                unwrapped_phase = normalize_array(unwrap_phase ( 2*np.pi*normalize_array( phase_compensation(phase) ) - np.pi ))
                # unwrapped_phase = phase_compensation(phase)


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
                
                    # --------------- SAVE INFO MIN MAX IN REGIONS
                    # --------------- SAVE MEAN of the fingermark contrast


                fingermark_folder = michelson_folder + f"/fingermark_{fingermark_counter}"
                print("fingermarkfolderdebugging",fingermark_folder)
                os.makedirs(fingermark_folder, exist_ok=True)
                path_for_amplitude  = fingermark_folder + f"/amplitude_fingermark_{fingermark_counter}.png"
                path_for_phase      = fingermark_folder + f"/phase_fingermark_{fingermark_counter}.png"

                if save_info:
                    save_minmax_txt(list_of_mins_maxs_amplitude, fingermark_folder + f'/contrast_amplitude{fingermark_counter}.txt')
                    save_minmax_txt(list_of_mins_maxs_phase, fingermark_folder + f'/contrast_phase{fingermark_counter}.txt')

                amplitude_from_complex, unwrapped_phase = np.uint8(255*normalize_array(amplitude_from_complex)), np.uint8(255*normalize_array(phase))

                if save_info:
                    cv2.imwrite(path_for_amplitude,overlay_amplt)
                    cv2.imwrite(path_for_phase, overlay_phase)

                number_of_fingermarks_per_image -=1
                fingermark_counter +=1

            # Here is saved the mean of the fingermaks
            if save_info:
                save_txt(y_data=fingerprints_amplitude_contrast, file_name= michelson_folder + f"/ampl_contrast.txt")
                save_txt(y_data=fingerprints_phase_contrast,     file_name= michelson_folder + f"/phase_contrast.txt")
                

                #   re-save csv
                df_amp, df_phase = collect_fingermark_summaries(michelson_folder)
                amp_csv, phase_csv = save_outputs(michelson_folder, df_amp, df_phase)
                print("\n modualtedmap:");print(df_amp.head())
                print("\n phase:");print(df_phase.head())\

                # Comprobar si es un archivo
                if os.path.isfile( michelson_folder+"/grading.csv"):
                    print(" ")
                else:
                    print("the file for unil grading doesnot exist.")
                    with open("vacio.csv", "w") as archivo:
                        pass  # No hace nada, solo lo crea

                # full plot 
                try:
                    ampl_phase_plot.make_csv_plot(amp_csv, phase_csv, michelson_folder+"/grading.csv")
                except:
                    pass

