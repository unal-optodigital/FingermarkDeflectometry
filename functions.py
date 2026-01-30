import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
import vmbpy
from vmbpy import *
from screeninfo import get_monitors
import sys
import cv2
from queue import Queue, Empty
import os
import time
from PIL import Image
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import math
from matplotlib.widgets import TextBox, Button

# mpl.rcParams['toolbar'] = 'none'


##-- global variables for camera and vimba context
cam = None                  
vmb = None                  

def fringe_generator(freq_ini: int, screen_width: int, screen_height: int):
    """
    fringe generator, select the frequency, angle and returns 
    a list of arrays that corresponds to individual phase-shifts
    of the generated pattern. Additionally the sixth image is for uniform illumination
    """
    phase_shift = [0, np.pi/2, np.pi, 3*np.pi/2,2*np.pi, 1]     # phase values 
    angle_ini = 0      # initial angle
     
    #  Meshgrid related to the resolution of the second screen to project the structured-pattern 
    x = np.linspace(-5, 5, screen_width)
    y = np.linspace(-5, 5, screen_height)
    X, Y = np.meshgrid(x, y)

    # initial rotation of the fringes 
    initial_rotation = X * np.cos(angle_ini) + Y * np.sin(angle_ini)

    # figure features
    fig_patron, ax_patron = plt.subplots(figsize=(15,7))
    fig_patron.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax_patron.axis('off')
    manager = plt.get_current_fig_manager()
    ax_patron.set_position([0, 0, 1, 1])  # hacer que el eje ocupe toda la figura (ventana)

    # figure + sliders
    fig_slider, (ax_freq, ax_angle) = plt.subplots(2, 1, figsize=(3, 2))
    plt.subplots_adjust(left=0.2, bottom=0.3, top=0.95)

    # dislpay initial pattern
    im = ax_patron.imshow( np.sin(2 * np.pi * freq_ini * initial_rotation), cmap='gray', extent=[min(x), max(x), min(y), max(y)], aspect='auto',vmin=-0.8,vmax=0.8)
    
    # slider features
    slider_freq = Slider(ax_freq, 'Frequency', valmin = 0.0, valmax = 5, valinit = freq_ini, valstep = 0.05)
    slider_angle = Slider(ax_angle, 'Rotation', valmin = 0, valmax = 2*np.pi, valinit = angle_ini, valstep = np.pi / 4)

    def update_pattern(phase_value):
        freq_updated = slider_freq.val        
        angle_updated = slider_angle.val    

        # updated pattern when a phase-shift has been applied
        if phase_value != 1:
            coord_rotadas = X * np.cos(angle_updated) + Y * np.sin(angle_updated)
            patron_updated = np.sin(2 * np.pi * freq_updated * coord_rotadas + phase_value)
        
        # uniform illumination
        else:
            coord_rotadas = X * np.cos(angle_updated) + Y * np.sin(angle_updated)
            patron_updated = np.sin( 0 * coord_rotadas + np.pi/4)

        # update de plot
        try:
            im.set_data(patron_updated)
            fig_patron.canvas.draw()
            fig_patron.canvas.flush_events()
        except Exception as e:
            pass  

        return patron_updated
    
    # calls the update_pattern function when sliders changes
    slider_freq.on_changed(update_pattern)
    slider_angle.on_changed(update_pattern)

    plt.show()  
                

    # list to store de fringe patterns to project
    projected_pattern = []

    for shifting in phase_shift:
        
        shifted_fringe = update_pattern(shifting)

        #  to convert in 8-bits image
        if shifting != 1:
            shifted_fringe = np.uint8(((shifted_fringe-np.min(shifted_fringe))/np.max(shifted_fringe-np.min(shifted_fringe)))*255)
        
        # store array in list
        projected_pattern.append(shifted_fringe)

    return projected_pattern

def plot_arrays(*arrays,
                 titles=None,
                 cmap="viridis",
                 cmaps=None,
                 origin="upper",
                 ncols=None,
                 figsize=None,
                 colorbar=False,
                 same_scale=False,
                 vmin=None,
                 vmax=None,
                 sharex=True,
                 sharey=True,
                 aspect="auto",
                 plot = True):
    """
    Muestra uno o varios arrays con imshow.
    - Si se pasa un solo array: una figura, un eje.
    - Si se pasan varios arrays: una figura con subplots.

    Soporta:
    - 2D: se muestra con cmap (por defecto 'gray')
    - 3D (H,W,3) o (H,W,4): se interpreta como RGB/RGBA (cmap se ignora)
    """

    # Permite pasar una lista/tupla única: imshow_multi([A,B,C])
    if len(arrays) == 1 and isinstance(arrays[0], (list, tuple)) and not isinstance(arrays[0], np.ndarray):
        arrays = tuple(arrays[0])

    if len(arrays) == 0:
        raise ValueError("Debes pasar al menos un array.")

    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs)

    # Validaciones y normalización mínima
    for i, a in enumerate(arrs):
        if a.ndim not in (2, 3):
            raise ValueError(f"Array #{i} debe ser 2D o 3D (RGB/RGBA). ndim={a.ndim}")
        if a.ndim == 3 and a.shape[-1] not in (3, 4):
            raise ValueError(f"Array #{i} 3D debe tener canales 3 o 4. shape={a.shape}")
        # Aviso suave si hay NaN/inf (no detiene)
        if np.issubdtype(a.dtype, np.floating):
            if not np.isfinite(a).all():
                # No imprimo por defecto; puedes activar si quieres
                pass

    # Config de colormaps por imagen
    if cmaps is None:
        cmaps = [cmap] * n
    else:
        if len(cmaps) != n:
            raise ValueError("Si hay 'cmaps', debe tener la misma longitud que arrays.")

    # Escala común opcional (solo para 2D)
    if same_scale:
        vals = []
        for a in arrs:
            if a.ndim == 2:
                vals.append(a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a.ravel())
        if len(vals) > 0:
            vals = np.concatenate([v.ravel() for v in vals])
            gvmin = np.min(vals) if vmin is None else vmin
            gvmax = np.max(vals) if vmax is None else vmax
        else:
            gvmin, gvmax = vmin, vmax
    else:
        gvmin, gvmax = vmin, vmax

    # Layout
    if ncols is None:
        ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    ims = []
    for k, (ax, a) in enumerate(zip(axes, arrs)):
        is_rgb = (a.ndim == 3)
        if same_scale and (not is_rgb):
            im = ax.imshow(a, cmap=cmaps[k], origin=origin, aspect=aspect, vmin=gvmin, vmax=gvmax)
        else:
            im = ax.imshow(a, cmap=None if is_rgb else cmaps[k], origin=origin, aspect=aspect, vmin=vmin, vmax=vmax)

        if titles is not None:
            ax.set_title(titles[k] if k < len(titles) else f"img {k}")
        else:
            ax.set_title(f"img {k}" if n > 1 else "")

        ax.set_axis_off()
        ims.append(im)

        if colorbar and (not is_rgb):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Apagar ejes sobrantes si la grilla es más grande que n
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if plot == True:
        plt.show()
    return fig, axes[:n], ims

def normalize_array(array = None):
    """
    input: an numpy array
    output: a normalized numpy array from 0 to 1
    """
    max = np.max(array)
    min = np.min(array)  
    
    if min == 0 and max == 0:
        print("array is a zero, NO NORMALIZABLE")
        return 0
    elif min < 0:
        array = array - min
        new_max = np.max(array)
        array = array/new_max
        return array
    else:
        return array/max
    
def phase_compensation(wrappedphase):
    """
    wrapped phase must be a 8-bit numpy array
    """     
    phase_image = np.double(wrappedphase)
    phase_image = phase_image.squeeze()
    phase_image = 2*np.pi*normalize_array(wrappedphase) - np.pi
    # phase_image = normalize_array(wrappedphase)

    print("shape",np.shape(phase_image))
    print("min max",np.min(phase_image), np.max(phase_image))

    # create a phasor to make the compensation
    complex_image = np.exp(2*np.pi*1j * phase_image)

    # create the meshgrid for the polynomial compensation
    x = np.linspace(-5, 5, phase_image.shape[1])
    y = np.linspace(-5, 5, phase_image.shape[0])
    X, Y = np.meshgrid(x, y)

    # slider values  for the polynomial coefficients 
    global slider_value1
    global slider_value2
    global slider_value3
    global slider_value4
    global slider_value5
    slider_value1=-0.015
    slider_value2=0
    slider_value3=0
    slider_value4=0
    slider_value5=0

    # creation of the polynomio
    def polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5):                  # FASE polynomial EN X,Y, 
        return slider_value1 * X + slider_value2*Y + slider_value3*X**2 + slider_value4 * X**2 + slider_value5

    # plots
    fig, ax = plt.subplots(figsize=(13, 8))                          
    ax.axis('off')
    fig.subplots_adjust(left=0, right=0.5, top=0.4, bottom=0.3)
    ax_original = plt.subplot(1, 1, 1)
    img_original = ax_original.imshow( np.angle(np.multiply(np.exp(1j * phase_image),np.exp(1j * 2 * np.pi * polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5) / 0.4))), extent=[-5, 5, -5, 5], cmap='gray')           # mapa de color de campo complejo
    ax_original.set_title("Compensación en X,Y")
    plt.tight_layout(pad=4.0)

    # sliders features
    slider_ax1 = plt.axes([0.2, 0.25, 0.03, 0.65])
    slider1 = Slider(slider_ax1, 'x', -1,1, valinit=0.05, valstep=0.0001,orientation='vertical')
    slider_ax5=plt.axes([0.15, 0.25, 0.03, 0.65])
    slider5=Slider(slider_ax5, 'offset', -2, 2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax2 = plt.axes([0.1, 0.25, 0.03, 0.65])
    slider2 = Slider(slider_ax2, 'y', -2,2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax3=plt.axes([0.8, 0.25, 0.03, 0.65])
    slider3=Slider(slider_ax3, 'x^2', -2, 2, valinit=0, valstep=0.001,orientation='vertical')
    slider_ax4=plt.axes([0.04, 0.25, 0.03, 0.65])
    slider4=Slider(slider_ax4, 'y^2', -2, 2, valinit=0, valstep=0.001,orientation='vertical')

 
 
    #update function when a slider changes
    def update(val):
        global slider_value1
        global slider_value2
        global slider_value3
        global slider_value5
        slider_value1 = slider1.val
        slider_value2 = slider2.val
        slider_value3 = slider3.val
        slider_value4 = slider4.val
        slider_value5 = slider5.val

        # evaluation of the polynomial compensation 
        updated_complex_image = np.multiply(np.exp(1j * phase_image),np.exp(1j * 2 * np.pi * polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5) / 0.4))
        img_original.set_data(np.angle(updated_complex_image)) # update the plot with the compensed polynomial
        fig.canvas.draw_idle()

    # sliders
    slider1.on_changed(update)       
    slider2.on_changed(update)
    slider3.on_changed(update)       
    slider4.on_changed(update)
    slider5.on_changed(update)
    plt.show()

    # print("coeff_values: ",slider_value1, slider_value2, slider_value3,slider_value4, slider_value5)

    # creation of the compensed phase with the adecuate polynomial coef values
    compensed = ((np.angle(np.multiply(np.exp(1j * phase_image),np.exp(1j * 2 * np.pi * polynomial(X, Y, slider_value1, slider_value2,slider_value3,slider_value4, slider_value5) / 0.4))))+np.pi)/(2*np.pi)*255
    print("min and max value of compensed phase",np.min(compensed), np.max(compensed))

    return compensed

def desenvolvimiento(wrapped_phase):
    """ phase unwrraping
    input: a wrapped phase array-8bits
    """
    # convertion to 8-bits
    image = np.double(wrapped_phase)
    image = np.uint8(image)

    # convertion to values between (-pi,pi]
    image_wrapped = (image/255)*2*np.pi-np.pi   
    
    # to computate the unwrapping
    image_unwrapped = unwrap_phase(image_wrapped)

    # display the unwrapping array
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(14,9))
    ax1.imshow(wrapped_phase, cmap='gray')
    ax1.set_title('Original phase')
    ax2.imshow(image_unwrapped, cmap='gray')
    ax2.set_title('Unwrapped phase')
    plt.show()

    return image_unwrapped

def phasor(array_of_frames):
    """
    creation of phasor from array of frames (I0-I2)+j(I3-I1)
    or making the redundance operation with five arrays: I0+I4-2*I2  + 2*j(I3-I1)
    """
    #normalización del array
    normalization = [(im/(np.max(im))) for im in array_of_frames] 

    # Real part
    parte_real = (normalization[0] + normalization[4] - 2*normalization[2])

    # imag part
    parte_imaginaria = 2*1j*(normalization[3]-normalization[1])

    return parte_real + parte_imaginaria

def amplitude_from_phasor(complex_phasor):  
    """
    modulated_intensity map from complex phasor
    """
    # abs value of a+ib --> sqrt(b^2+a^2)
    amplitude=((np.abs(complex_phasor))/(np.max(np.abs(complex_phasor))))*255
    return amplitude

def amplitude_from_frames(array_of_frames):  
    """
    Modulated intensity map from frames
    """
    #normalization
    normalization = [(im/(np.max(im))) for im in array_of_frames]
    shifted13 = normalization[1]-normalization[3]
    shifted02 = normalization[0]-normalization[2]
    squared  = shifted13**2 + shifted02**2
    amplitude = np.sqrt(squared) 
    amplitude=((amplitude)/(np.max(amplitude)))*255
    return amplitude

def phase_calculation_from_complex_phasor(complex_phasor): 
    """
    for obtaining the phase is required an array of the complex phasor 
    """
    #Computation of wrapped phase with arctan2 that returns a matrix with values bewteen [-pi,pi)
    wrapped_phase = np.arctan2(np.imag(complex_phasor),np.real(complex_phasor))
    # convertion to 8-bits
    wrapped_phase_8_bits = (((wrapped_phase + np.pi)/(2*np.pi))*255)

    return wrapped_phase_8_bits

def phase_calculation_from_array(array_of_frames):
    """
    Phase obtantion from the frames. A list of frames is required
    """
    # normalization of the captures. Now, the values are between [0,1]
    normalization=[(im/(np.max(im))) for im in array_of_frames] 

    # Real array
    parte_real = (normalization[0]+normalization[4]-2*normalization[2])

    # Imag array
    parte_imaginaria = 2*(normalization[3]-normalization[1])

    # Convertion 8-bits
    parte_real = np.uint8( parte_real/np.max(parte_real)  )
    parte_imaginaria = np.uint8(( parte_imaginaria / np.max(parte_imaginaria) ))

    # phase [-pi,pi) values
    phase = np.arctan2((normalization[1]-normalization[3]),(normalization[0]-normalization[2]))

    # Convertion 8-bits
    phase = np.uint8(((phase+np.pi)/(2*np.pi))*255)

    return phase

def monitor_info():
    """
    displacement, resize_height, resize_width
    Get the main and second monitor info for resizing and moving the 
    fringe pattern to the second screen
    """

    monitors = get_monitors()
    number_of_monitors = int(np.shape(monitors)[0])

    for i in range(0,number_of_monitors):
        if monitors[i].is_primary:
            main_monitor = monitors[i]
        else:
            second_monitor = monitors[i]
    window_displacement = int(main_monitor.width)
    try:
        resize_window_height = int(second_monitor.height)
        resize_window_width = int(second_monitor.width)
        
       
    except NameError:
        print('second screen not detected')
        sys.exit(1)          # close the code
    
    return window_displacement, resize_window_height, resize_window_width
        
def positionate_camera():
    """
    calibrate camera  position
    """
    WIN = 'camera'
    q = Queue(maxsize=1) 
    running = True

    def frame_handler(cam: Camera, stream: Stream, frame: Frame):
        """Callback de VmbPy: convertir a Mono8, pasar a OpenCV y reencolar el frame."""
        try:
           
            frame.convert_pixel_format(PixelFormat.Mono8)
            img = frame.as_opencv_image().copy()  
            
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(img)
        finally:
            stream.queue_frame(frame)  # to continue receiving frames

    with VmbSystem.get_instance() as vmb:           # context of vimba
        cams = vmb.get_all_cameras()
        if not cams:
            raise RuntimeError("not detected cameras.")
        with cams[0] as cam:                        # opens the camera in the vimba context
    
            cam.start_streaming(frame_handler)

            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL) # create de window once

            try:
                while True:
                    # close the window 
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    try:
                        img = q.get(timeout=0.01)  # wait for the frame
                        cv2.imshow(WIN, img)
                    except Empty:
                        # continues if the frames is not updated
                        pass

                    # ESC for continue and close 
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
            finally:
                cam.stop_streaming()
                cv2.destroyAllWindows()

def init_camera(exposure_time: int = 50000, 
                gain: int = 0, 
                pixel_format: str = 'Mono8'):
    '''
    Initialize camera with exposure_time, gain, and pixel format.
    '''
    global cam
    global vmb
    if vmb is None:
        vmb = vmbpy.VmbSystem.get_instance()  # INITIALIZE VIMBA CONTEXT
        vmb.__enter__()
        cams = vmb.get_all_cameras()  # LIST OF CAMERAS
        cam = cams[0]  # SELECT THE FIRST CAMERA
        cam.__enter__()

        # ENABLE THE FEATURES CONTROL
    
        fr_enable = cam.get_feature_by_name('AcquisitionFrameRateEnable')
        fr_enable.set(True)
       
        # if exposure_time !=0:
        #     cam.get_feature_by_name('ExposureAuto').set('Off')
        #     cam.get_feature_by_name('ExposureTime').set(exposure_time)  # microseg---> 
        # else:
        #     cam.get_feature_by_name('ExposureAuto').set('On')

        #cam.get_feature_by_name('GainAuto').set('Off')
        cam.get_feature_by_name('Gain').set(gain)  
        #cam.get_feature_by_name('BalanceWhiteAuto').set('Off')
        if pixel_format == 'Mono8':
            cam.set_pixel_format(PixelFormat.Mono8) 
        elif pixel_format == 'RGB8':
            cam.set_pixel_format(PixelFormat.Rgb8)

        frame_rate = cam.get_feature_by_name('AcquisitionFrameRate')
        max_posible_frame_rate = frame_rate.get_range()[1]
        try:
            frame_rate.set(max_posible_frame_rate)
        except:
            pass

def close_camera():
    '''
    The camera closes correctly.
    '''
    global cam
    global vmb
    #cam.set_pixel_format(PixelFormat.Mono8)
    cam.__exit__(None, None, None)
    vmb.__exit__(None, None, None)

def create_folder(path_folder: str):
    """string with the path of the newfolder"""
    # Create the folder if doesn't exist 
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

def phase_shifting_loop(cam, vmb, patterns_to_project, path_folder):
    
    """
    Loop where the acquisition of shifted-pattern frames will happen
    """
    print("Frames acquisition has started...")

    delay_camera_screen  = 500E-3   # delay to sinchronize screen projection and camera acquisition
    shifted_frames = []             # List for storing the acquired frames
    fringe_shift_counter = 0        # fringe shift counter to guarantee the phase shifting-technique
    while True:
        cv2.imshow('pattern',((patterns_to_project[fringe_shift_counter])))
        cv2.waitKey(1)
        time.sleep(delay_camera_screen)
            
        # acquiring and saving the shifted-phase
        cam.set_pixel_format(PixelFormat.Mono8) 
        frame = cam.get_frame()
        frame = frame.as_numpy_ndarray()
        frame = np.squeeze(frame)
  

        # storing the frame for post-processing
        shifted_frames.append(frame)
        fringe_shift_counter +=1

        # finish at the sixth capture
        if fringe_shift_counter == 6:
            cam.set_pixel_format(PixelFormat.Mono8) 
            frame_color = cam.get_frame()
            frame_color = frame_color.as_numpy_ndarray()
            frame_color = np.squeeze(frame_color)
            shifted_frames.append(frame_color)
            cam.set_pixel_format(PixelFormat.Mono8)
            cv2.destroyAllWindows()
            break

    return shifted_frames

def save_8_bit_images(array_of_images, path_folder: str):
    """
    save phase [-1], 
    modulated intensity map[-2],
    color image of the surface pos[-3],
    save shifted frames[-4..-8],
    un path_folder
    """
    phase = array_of_images.pop()
    amplitude = array_of_images.pop()
    color_image = array_of_images.pop()
    
    if os.path.exists(path_folder):
        
        cv2.imwrite(path_folder+"/modulated_intensity_map.png", np.uint8(255*amplitude/np.max(amplitude)))
        cv2.imwrite(path_folder+"/wrapped_phase.png",phase)
        cv2.imwrite(path_folder+f"/color_image.png", color_image)

        for i in range(0,len(array_of_images)):
            cv2.imwrite(path_folder + f"/Phase_{i}.png", array_of_images[i])
        # realización de la compensacion
        # compensation = input("Realize the phase compensation (yes) (no): ")
        compensation = 'yes'
        if compensation.lower() == "yes":
            compensated_phase = phase_compensation(phase)
            plot_arrays([phase,compensated_phase, amplitude,color_image], ncols=2, cmap='gray')
            cv2.imwrite(path_folder+"/compensated_phase.png", np.uint8(255*compensated_phase/np.max(compensated_phase)))
    print("Images succesfully saved")

def run_camera_and_fringes_ui(
    freq_ini: float,
    screen_width: int,
    screen_height: int,
    pattern_resize_w: int,
    pattern_resize_h: int,
    pattern_displacement_x: int = 1920,   # move 1920 pixels the second screen
    pattern_displacement_y: int = 0,
    slider_fig_size=(5, 2)
):
    """
    -Slider(freq, angle of fringes in main window)
    -Projected fringes on second screen
    -OpenCV visualization of the camera 
    -Close any window o press ESC and the code will continue
    -returns the list of projected pattern to display and make the phase shift
    -5 sinusoidal patterns with phase: [0, π/2, π, 3π/2, 2π]  and 1 for uniform illumination
    """

    # geometrý 
    # Meshgrid
    x = np.linspace(-5, 5, screen_width)
    y = np.linspace(-5, 5, screen_height)
    X, Y = np.meshgrid(x, y)

    # sliders state
    state = {
        'freq': float(freq_ini),
        'angle': 0.0,
        'exposure_time': 20000.0,  # µs initial value (20 ms)
        'stop': False,
        'cam': None,               
    }

    #  slider fig (Matplotlib)
    fig_sliders, (ax_freq, ax_angle, ax_exp) = plt.subplots(3, 1, figsize=slider_fig_size)
    plt.subplots_adjust(left=0.2, bottom=0.25, top=0.95, hspace=0.5)
    fig_sliders.canvas.manager.set_window_title('Fringe Controls')
    mgr = fig_sliders.canvas.manager
    try:
        mgr.window.wm_geometry("+0+0")  # TkAgg
    except Exception:
        mgr.window.move(1000, 0)           # Qt5/Qt6

    slider_freq = Slider(ax_freq, 'freq', valmin=0.0, valmax=10.0,
                         valinit=freq_ini, valstep=0.05)
    slider_angle = Slider(ax_angle, 'tilt', valmin=0.0, valmax=2*np.pi,
                          valinit=0.0, valstep=np.pi/4)
    
    # typical range
    EXP_MIN_US, EXP_MAX_US, EXP_STEP_US = 50.0, 1_000_000.0, 100.0
    slider_exp = Slider(ax_exp, 'ex_time [µs]', valmin=EXP_MIN_US, valmax=EXP_MAX_US,
                        valinit=state['exposure_time'], valstep=EXP_STEP_US)

    def _set_exposure_if_possible(cam_obj, exposure_us: float):
        """set ExposureTime/ExposureTimeAbs in µs."""
        if cam_obj is None:
            return
        # Clamp slider range
        exposure_us = float(np.clip(exposure_us, EXP_MIN_US, EXP_MAX_US))
        # autoexposure off
        for ename in ('ExposureAuto',):
            try:
                feat = cam_obj.get_feature_by_name(ename)
                try:
                    feat.set('Off')
                except Exception:
                    pass
            except Exception:
                pass
        # by feature
        for fname in ('ExposureTime', 'ExposureTimeAbs'):
            try:
                feat = cam_obj.get_feature_by_name(fname)
                feat.set(exposure_us)
                return
            except Exception:
                continue

    def on_change(_):
        state['freq'] = float(slider_freq.val)
        state['angle'] = float(slider_angle.val)
        state['exposure_time'] = float(slider_exp.val)
        if state['cam'] is not None:
            _set_exposure_if_possible(state['cam'], state['exposure_time'])

    slider_freq.on_changed(on_change)
    slider_angle.on_changed(on_change)
    slider_exp.on_changed(on_change)

    def on_close_matplotlib(event):
        state['stop'] = True
    fig_sliders.canvas.mpl_connect('close_event', on_close_matplotlib)

    # display the slider plot and do not block the code
    plt.show(block=False)

    #  WINDOWS OpenCV (pattern and camera) 
    WIN_PATTERN = 'pattern'
    cv2.namedWindow(WIN_PATTERN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PATTERN, pattern_resize_w, pattern_resize_h)
    cv2.moveWindow(WIN_PATTERN, pattern_displacement_x, pattern_displacement_y)

    # camera window
    WIN_CAMERA = 'camera'
    cv2.namedWindow(WIN_CAMERA, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WIN_CAMERA, 1000, 1000)  

    # asynchronous for vimba
    q = Queue(maxsize=1)

    def frame_handler(cam: Camera, stream: Stream, frame: Frame):
        """Callback corto: convierte a Mono8, copia numpy y reencola; siempre requeue."""
        try:
            frame.convert_pixel_format(PixelFormat.Mono8)
            img = frame.as_opencv_image().copy()
            # most recent frame
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(img)
        finally:
            stream.queue_frame(frame)

    # vimba context
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            plt.close(fig_sliders)
            cv2.destroyAllWindows()
            raise RuntimeError("Verify the camera conection.")

        with cams[0] as cam:
            # for slider changing in real time
            state['cam'] = cam

            # current exposure
            current_exp = None
            for fname in ('ExposureTime', 'ExposureTimeAbs'):
                try:
                    feat = cam.get_feature_by_name(fname)
                    current_exp = float(feat.get())
                    break
                except Exception:
                    continue
            if current_exp is not None:
                # in the range
                if EXP_MIN_US <= current_exp <= EXP_MAX_US:
                    slider_exp.set_val(current_exp)
                    state['exposure_time'] = current_exp
                else:
                    # initial value of exposure
                    _set_exposure_if_possible(cam, state['exposure_time'])
            else:
                # initial value
                _set_exposure_if_possible(cam, state['exposure_time'])

            # Init streaming
            cam.start_streaming(frame_handler)

            try:
                while not state['stop']:
                    # close if any window is closed
                    if cv2.getWindowProperty(WIN_CAMERA, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if cv2.getWindowProperty(WIN_PATTERN, cv2.WND_PROP_VISIBLE) < 1:
                        break

                    # ==== Camera ====
                    try:
                        img = q.get(timeout=0.01)
                        cv2.imshow(WIN_CAMERA, img)
                    except Empty:
                        pass

                    # ==== fringes
                    coord_rot = X * np.cos(state['angle']) + Y * np.sin(state['angle'])
                    patt = np.sin(2 * np.pi * state['freq'] * coord_rot)  # phase 0
                    # 8-bits
                    patt_u8 = ((patt - patt.min()) / (patt.max() - patt.min() + 1e-12) * 255.0).astype(np.uint8)
                    cv2.imshow(WIN_PATTERN, patt_u8)

                    # ==== events 
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break

                    plt.pause(0.001)

            finally:
                cam.stop_streaming()
                state['cam'] = None

    # close the cv2 windows and matplotlib ones
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        plt.close(fig_sliders)
    except Exception:
        pass

    # generation of patterns
    freq_final = state['freq']
    angle_final = state['angle']
    exposure_final_us = float(state['exposure_time'])  # 
    phase_shift = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, 1]  # el "1" indicates uniform illumination

    projected_pattern = []
    coord_rot_final = X * np.cos(angle_final) + Y * np.sin(angle_final)

    for shifting in phase_shift:
        if shifting != 1:
            pat = np.sin(2 * np.pi * freq_final * coord_rot_final + shifting)
            pat_u8 = np.uint8(((pat - pat.min()) / (pat.max() - pat.min() + 1e-12)) * 255.0)
            projected_pattern.append(pat_u8)
        else:
            # uniform illumination
            pat_uniform = np.sin(0 * coord_rot_final + np.pi/4)
            projected_pattern.append(pat_uniform)

    return projected_pattern, exposure_final_us

def load_image_gray(path: str):
    """
    load an image and returns a float arrray of 32 bits in gray scale
    even if the image is RGB
    """
    img = Image.open(path)
    arr = np.array(img)

    # Si es RGB/RGBA -> a gris (luma aproximada)
    if arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        arr = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    return arr.astype(np.float32)

def crop_image(img: np.ndarray) -> np.ndarray:
    roi_coords = None  
    roi_array  = None  

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    ax0.set_title("ROI")
    ax1.set_title("Crop (ROI)")
    ax0.imshow(img, cmap="gray")
    im_crop = ax1.imshow(img, cmap="gray")
    ax0.set_axis_off(); ax1.set_axis_off()

    def onselect(eclick, erelease):
        nonlocal roi_coords, roi_array
        
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        h, w = img.shape
        x0 = max(0, min(w-1, x0))
        x1 = max(1, min(w,   x1))
        y0 = max(0, min(h-1, y0))
        y1 = max(1, min(h,   y1))

        roi_coords = (x0, x1, y0, y1)
        roi_array = img[y0:y1, x0:x1]

        im_crop.set_data(roi_array)
        ax1.set_xlim(0, roi_array.shape[1])
        ax1.set_ylim(roi_array.shape[0], 0)
        fig.canvas.draw_idle()

        print(f"ROI coords: x0={x0}, x1={x1}, y0={y0}, y1={y1} | shape={roi_array.shape}")

    rs = RectangleSelector(
        ax0, onselect,
        useblit=True,
        button=[1],          # click izquierdo
        interactive=True,
        spancoords="pixels"
    )

    plt.tight_layout()
    plt.show()

    # Cuando cierres la ventana, si seleccionaste algo, aquí queda disponible:
    if roi_coords is not None:
        x0, x1, y0, y1 = roi_coords

    return roi_coords, roi_array

def partition_in_boxes(h: int, w: int, cols_per_row=(5,5,5,5,5)):
    """
    Divide una imagen (h,w) en cajas según cols_per_row.
    Para 3x3: cols_per_row=(3,3,3) -> 9 regiones.
    Retorna lista de bounds (y0,y1,x0,x1) en orden row-major.
    """
    cols_per_row = list(cols_per_row)
    if any(c < 1 for c in cols_per_row):
        raise ValueError("cols_per_row inválido")
    nrows = len(cols_per_row)

    base_h, rem_h = divmod(h, nrows)
    row_heights = [base_h + (1 if i < rem_h else 0) for i in range(nrows)]

    regions = []
    y = 0
    for r, ncols in enumerate(cols_per_row):
        y0, y1 = y, y + row_heights[r]
        y = y1

        base_w, rem_w = divmod(w, ncols)
        col_widths = [base_w + (1 if j < rem_w else 0) for j in range(ncols)]

        x = 0
        for cw in col_widths:
            x0, x1 = x, x + cw
            regions.append((y0, y1, x0, x1))
            x = x1

    return regions

def region_min_max(img: np.ndarray, cols_per_row=(3, 2)):
    """Calculate min/max and global coords (y,x) in  each box"""
    if img.ndim != 2:
        raise ValueError("img debe ser 2D (grises).")

    h, w = img.shape
    regs = partition_in_boxes(h, w, cols_per_row=cols_per_row)

    out = []
    for i, (y0, y1, x0, x1) in enumerate(regs):
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        py_min, px_min = np.unravel_index(int(np.argmin(patch)), patch.shape)
        py_max, px_max = np.unravel_index(int(np.argmax(patch)), patch.shape)

        # Global
        y_min, x_min = y0 + py_min, x0 + px_min
        y_max, x_max = y0 + py_max, x0 + px_max

        out.append({
            "region_id": i,
            "bounds": (y0, y1, x0, x1),
            "min_val": float(img[y_min, x_min]),
            "min_pos": (int(y_min), int(x_min)),
            "max_val": float(img[y_max, x_max]),
            "max_pos": (int(y_max), int(x_max)),
        })

    return out
    
def plot_regions_with_mins_and_maxs(img: np.ndarray, info_list) -> list[list[float]]:
    """
    Dibuja las regiones (bounds) y marca min/max (posiciones ya vienen en info_list).
    Además retorna: [[min,max], [min,max], ...] en el mismo orden de info_list.

    Requiere que cada dict d tenga:
      d["bounds"] = (y0,y1,x0,x1)
      d["min_pos"] = (y_min,x_min)
      d["max_pos"] = (y_max,x_max)
    (region_id es opcional para el label)
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_title("Regions + min/max local")
    ax.set_axis_off()

    out = []
    for d in info_list:
        y0, y1, x0, x1 = d["bounds"]

        # draw box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2))

        # mark min/max
        y_max, x_max = d["max_pos"]
        y_min, x_min = d["min_pos"]
        ax.plot(x_max, y_max, marker="x", markersize=10, linewidth=2)
        ax.plot(x_min, y_min, marker="o", markersize=6)

        # label (si existe)
        rid = d.get("region_id", None)
        if rid is not None:
            ax.text(x0 + 3, y0 + 15, f"R{rid}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.6))

        # min/max values (desde la imagen, usando las posiciones)
        out.append([float(img[y_min, x_min]), float(img[y_max, x_max])])

    plt.tight_layout()
    plt.show()

    return out  

def michelson_contrast_per_region(minmax_list: list[list[float]]) -> list[float]:
    """
    minmax_list: [[min,max],[min,max],...]
    retorna: [C0, C1, ...] con C = (Imax - Imin)/(Imax + Imin)
    Si Imax+Imin == 0 -> NaN
    """
    out = []
    for mn, mx in minmax_list:
        den = mx + mn
        out.append(float(np.abs((mx - mn)) / den) if den != 0 else float(0))

    return out


def segmentation_of_regions(
    img: np.ndarray,
    nrows: int = 4,
    ncols: int = 10,
    radius: int = 1,
    stat: str = "mean",       # "mean" | "median" | "mode"
    bins: int = 256,
    return_overlay: bool = False
):
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows y ncols deben ser >= 1")
    if radius < 0:
        raise ValueError("radius debe ser >= 0")
    if stat not in ("mean", "median", "mode"):
        raise ValueError("stat debe ser: 'mean', 'median' o 'mode'")
    if bins < 2:
        raise ValueError("bins debe ser >= 2")

    # a gris 2D
    if img.ndim == 3:
        rgb = img[..., :3].astype(np.float32)
        a = 0.299*rgb[..., 0] + 0.587*rgb[..., 1] + 0.114*rgb[..., 2]
    else:
        a = img.astype(np.float32) if not np.issubdtype(img.dtype, np.floating) else img

    H, W = a.shape
    xs = np.linspace(0, W, ncols + 1, dtype=int)
    ys = np.linspace(0, H, nrows + 1, dtype=int)

    nreg = nrows * ncols
    vals = [[] for _ in range(nreg)]
    pos  = [[None, None] for _ in range(nreg)]  # <- clicks: (y,x) para pick0 y pick1
    k = [0] * nreg

    def bid(x, y):
        c = np.searchsorted(xs, x, side="right") - 1
        r = np.searchsorted(ys, y, side="right") - 1
        return r * ncols + c if 0 <= r < nrows and 0 <= c < ncols else None

    def mode_value(patch: np.ndarray) -> float:
        if np.issubdtype(patch.dtype, np.integer):
            u, cnt = np.unique(patch, return_counts=True)
            return float(u[np.argmax(cnt)])
        p = patch.astype(np.float32)
        mn, mx = float(p.min()), float(p.max())
        if mx == mn:
            return mn
        q = np.floor((p - mn) / (mx - mn) * (bins - 1)).astype(np.int32)
        hist = np.bincount(q.ravel(), minlength=bins)
        b = int(np.argmax(hist))
        return float(mn + (b + 0.5) * (mx - mn) / bins)

    def local_value(x, y):
        if radius == 0:
            return float(a[y, x])
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        patch = a[y0:y1, x0:x1]
        if stat == "mean":
            return float(patch.mean())
        if stat == "median":
            return float(np.median(patch))
        return mode_value(patch)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(a, cmap="gray", interpolation="nearest")
    ax.set_title(f"Click: picks por región | stat={stat} radius={radius} | Cierra para terminar")
    ax.set_axis_off()

    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2))
            ax.text(x0 + 3, y0 + 15, f"R{i}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.6))

    mark0 = [ax.plot([], [], "o", markersize=6)[0] for _ in range(nreg)]
    mark1 = [ax.plot([], [], "x", markersize=10, linewidth=2)[0] for _ in range(nreg)]

    def refresh_region(i):
        if pos[i][0] is not None:
            y, x = pos[i][0]; mark0[i].set_data([x], [y])
        else:
            mark0[i].set_data([], [])
        if pos[i][1] is not None:
            y, x = pos[i][1]; mark1[i].set_data([x], [y])
        else:
            mark1[i].set_data([], [])
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or e.xdata is None or e.ydata is None:
            return
        x, y = int(e.xdata), int(e.ydata)
        if not (0 <= x < W and 0 <= y < H):
            return

        i = bid(x, y)
        if i is None:
            return

        v = local_value(x, y)
        j = len(vals[i]) if len(vals[i]) < 2 else k[i]

        if len(vals[i]) < 2:
            vals[i].append(v)
        else:
            vals[i][j] = v
            k[i] ^= 1

        pos[i][j] = (y, x)
        refresh_region(i)
        print(f"region={i} values={vals[i]}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    out = [[min(v), max(v)] if len(v) == 2 else [np.nan, np.nan] for v in vals]

    if not return_overlay:
        return out, pos

    fig.canvas.draw()
    overlay = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out, pos, overlay

def apply_clicks_to_image(
    img: np.ndarray,
    pos,                   # [[(y,x) or None, (y,x) or None], ...] len = nrows*ncols
    nrows: int,
    ncols: int,
    radius: int = 1,
    stat: str = "mean",
    bins: int = 256,
    return_overlay: bool = True
):
    if img.ndim == 3:
        rgb = img[..., :3].astype(np.float32)
        a = 0.299*rgb[..., 0] + 0.587*rgb[..., 1] + 0.114*rgb[..., 2]
    else:
        a = img.astype(np.float32) if not np.issubdtype(img.dtype, np.floating) else img

    H, W = a.shape
    xs = np.linspace(0, W, ncols + 1, dtype=int)
    ys = np.linspace(0, H, nrows + 1, dtype=int)

    def mode_value(patch: np.ndarray) -> float:
        if np.issubdtype(patch.dtype, np.integer):
            u, cnt = np.unique(patch, return_counts=True)
            return float(u[np.argmax(cnt)])
        p = patch.astype(np.float32)
        mn, mx = float(p.min()), float(p.max())
        if mx == mn:
            return mn
        q = np.floor((p - mn) / (mx - mn) * (bins - 1)).astype(np.int32)
        hist = np.bincount(q.ravel(), minlength=bins)
        b = int(np.argmax(hist))
        return float(mn + (b + 0.5) * (mx - mn) / bins)

    def sample_at(y, x) -> float:
        if radius == 0:
            return float(a[y, x])
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        patch = a[y0:y1, x0:x1]
        if stat == "mean":
            return float(patch.mean())
        if stat == "median":
            return float(np.median(patch))
        return mode_value(patch)

    # calcular [[min,max], ...] usando las dos posiciones por región
    out = []
    for p0, p1 in pos:
        if p0 is None or p1 is None:
            out.append([np.nan, np.nan])
            continue
        v0 = sample_at(p0[0], p0[1])
        v1 = sample_at(p1[0], p1[1])
        out.append([min(v0, v1), max(v0, v1)])

    if not return_overlay:
        return out

    # overlay (misma estética)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(a, cmap="gray", interpolation="nearest")
    ax.set_axis_off()
    ax.set_title("Aplicado desde clicks guardados")

    for r in range(nrows):
        for c in range(ncols):
            i = r * ncols + c
            x0, x1 = xs[c], xs[c + 1]
            y0, y1 = ys[r], ys[r + 1]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2))
            ax.text(x0 + 3, y0 + 15, f"R{i}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", alpha=0.6))

            if i < len(pos):
                if pos[i][0] is not None:
                    y, x = pos[i][0]
                    ax.plot(x, y, "o", markersize=6)
                if pos[i][1] is not None:
                    y, x = pos[i][1]
                    ax.plot(x, y, "x", markersize=10, linewidth=2)

    plt.tight_layout()
    fig.canvas.draw()
    overlay = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return out, overlay



def mean(list: list[float]):
    """
    Docstring for mean
    
    :param list: Description
    """
    sum_ = np.sum(list)
    mean = sum_/len(list)
    return mean 

def save_txt(x_data=None, y_data=None, file_name:str=None):
    """
    Docstring for save_txt
    
    :param x_data: x-axis data
    :param y_data: y-axis plot data
    :param file_name: 'name.txt'
    :type file_name: str
    """
    W = len(y_data)

    if x_data == None:
        x_data = np.arange(0,W,1)
    with open(file_name, 'w') as f:
        f.write("X_Pos\tY_Value\n") 
        for xi, yi in zip(x_data,y_data ):
            f.write(f"{xi:.4f}\t{yi:.4f}\n")

def imshow_with_textbox_ok(img: np.ndarray, label: str = "Valor:", initial: str = "1"):
    """
    Muestra un imshow + un TextBox para escribir un número + un botón OK que cierra.
    Retorna el valor (float) al cerrar.
    """

    state = {"val": None}

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
    ax.set_title('Select the amount of fingermarks in the image')
    ax.set_axis_off()

    tb_ax  = fig.add_axes([0.15, 0.02, 0.55, 0.06])  # [left, bottom, width, height]
    ok_ax  = fig.add_axes([0.73, 0.02, 0.12, 0.06])

    tb = TextBox(tb_ax, label, initial=initial)
    btn = Button(ok_ax, "OK")

    def parse(text):
        state["val"] = float(text)

    def on_ok(event):
        # si nunca dio Enter, intenta parsear lo que está escrito
        if state["val"] is None:
            try:
                parse(tb.text)
            except Exception:
                state["val"] = None
        plt.close(fig)

    tb.on_submit(parse)      # Enter en la caja
    btn.on_clicked(on_ok)    # click en OK

    plt.show()
    return state["val"]

def save_minmax_txt(listas, txt_path):
    """
    listas: [[min,max], [min,max], ...]
    Guarda en txt:
    region 1: min=..., max=...
    region 2: min=..., max=...
    ...
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, (mn, mx) in enumerate(listas, start=1):
            f.write(f"region {i}: min={mn}, max={mx}\n")