import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams['toolbar'] = 'None'  # quitar barra de herramientas de matplotlib

# Parámetros iniciales
f_init = 1.0
p_init = 0.0
angle_init = 0.0

# Resolución de la malla
resolution = 1500
x = np.linspace(-5, 5, resolution)
y = np.linspace(-5, 5, resolution)
X, Y = np.meshgrid(x, y)

# Crear la figura principal y los ejes para el mapa de colores
fig_main = plt.figure(figsize=(10, 6))
ax_main = fig_main.add_axes([0, 0, 1, 1])  # Usar add_axes para ocupar todo el espacio sin márgenes
#manager = plt.get_current_fig_manager()
#manager.full_screen_toggle()  # Pantalla completa

# Figura de los deslizadores
fig_sliders, (ax_fx, ax_p, ax_angle) = plt.subplots(3, 1, figsize=(4, 2))
plt.subplots_adjust(left=0.2, bottom=0.3, top=0.95, hspace=0)  # Espaciado de los deslizadores

# Crear sliders para fx, p y ángulo
slider_fx = Slider(ax_fx, 'Freq', 0, 15, valinit=8, valstep=0.2)
slider_p = Slider(ax_p, 'Desfase', 0, 3 * np.pi / 2, valinit=p_init, valstep=np.pi / 2)
slider_angle = Slider(ax_angle, 'Ángulo', 0, 360, valinit=135,valstep = 45)

# Inicializar el gráfico de la imagen
def imagen(fx, p, angle):
    theta = np.radians(angle)
    X_rot = X * np.cos(theta) - Y * np.sin(theta)
    Y_rot = X * np.sin(theta) + Y * np.cos(theta)
    Z = 10 * np.sin(2 * np.pi * fx * (X_rot + Y_rot) + p)
    return Z

Z = imagen(f_init, p_init, angle_init)
image = ax_main.imshow(Z, extent=[min(x), max(x), min(y), max(y)], cmap='gray', aspect='auto')
ax_main.axis('off')  # Ocultar los ejes
#ax_main.set_xticks([])  # Eliminar marcas de los ejes x
#ax_main.set_yticks([])  # Eliminar marcas de los ejes y

# Función para actualizar la imagen cuando cambian los parámetros
def update(val):
    fx = slider_fx.val
    p = slider_p.val
    angle = slider_angle.val
    Z = imagen(fx, p, angle)
    image.set_data(Z)
    fig_main.canvas.draw_idle()

# Vincular los deslizadores con la función de actualización
slider_fx.on_changed(update)
slider_p.on_changed(update)
slider_angle.on_changed(update)

plt.show()