import pygame
import tkinter as tk
import numpy as np
from numpy import pi, sin, cos, sqrt, arctan2
from math import *
import numpy as np
from numpy import sin, cos, sqrt, arctan2
from decimal import Decimal, ROUND_HALF_UP
import math

# Colores disponibles:

WHITE = (255,255,255)
RED = (255,0,0)
BLACK = (0,0,0)

def entradas(ache):

    def round_well(num):
        return Decimal(num).quantize(0, ROUND_HALF_UP)

    def dotproduct(v1, v2):
        return sum((a*b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

    # DATOS DE ENTRADA

    # Posición Punto 1
    P1 = np.array([P1x, P1y])

    # Posición Punto 2
    P2 = np.array([P2x, P2y])

    # Posición Punto 3
    P3 = np.array([P3x, P3y])

    # Posicion de los pivotes O2 y O4
    O2 = np.array([O2x, O2y])
    O4 = np.array([O4x, O4y])

    #########################################################
    # CALCULOS BASICOS

    # Vectores distancia entre los puntos 1, 2 y 3
    P21 = P2 - P1
    P31 = P3 - P1

    # Magnitudes de los vectores
    M_P21 = sqrt(P21[0]**2 + P21[1]**2)
    M_P31 = sqrt(P31[0]**2 + P31[1]**2)
    
    # Angulos entre los vectores P21, P31 y la horizontal
    delta2 = np.degrees(arctan2(P21[1],P21[0]))  #P21
    delta3 = np.degrees(arctan2(P31[1],P31[0]))  #P31

    # Vectores radio R1, R2 y R3
    R1 = P1 - O2
    R2 = P2 - O2
    R3 = P3 - O2

    # Magnitudes de R1, R2 y R3
    M_R1 = sqrt(R1[0]**2 + R1[1]**2)
    M_R2 = sqrt(R2[0]**2 + R2[1]**2)
    M_R3 = sqrt(R3[0]**2 + R3[1]**2)

    # Angulos de R1, R2 y R3
    zeta1 = np.degrees(arctan2(R1[1], R1[0])) # Angulo de R1
    zeta2 = np.degrees(arctan2(R2[1], R2[0])) # Angulo de R2
    zeta3 = np.degrees(arctan2(R3[1], R3[0])) # Angulo de R3

    # Angulos a calcular
    beta2 = 0
    beta3 = 0

    #########################################################
    # CALCULOS PARA BETA2 Y BETA3

    # Calculo de las constantes C
    C1 = M_R3*cos(np.radians(alpha2 + zeta3)) - M_R2*cos(np.radians(alpha3 + zeta2))
    C2 = M_R3*sin(np.radians(alpha2 + zeta3)) - M_R2*sin(np.radians(alpha3 + zeta2))
    C3 = M_R1*cos(np.radians(alpha3 + zeta1)) - M_R3*cos(np.radians(zeta3))
    C4 = -M_R1*sin(np.radians(alpha3 + zeta1)) + M_R3*sin(np.radians(zeta3))
    C5 = M_R1*cos(np.radians(alpha2 + zeta1)) - M_R2*cos(np.radians(zeta2))
    C6 = -M_R1*sin(np.radians(alpha2 + zeta1)) + M_R2*sin(np.radians(zeta2))

    # Calculo de las constantes A
    A1 = -(C3**2) - (C4**2)
    A2 = C3*C6 - (C4*C5)
    A3 = -(C4*C6) - (C3*C5)
    A4 = C2*C3 + C1*C4
    A5 = C4*C5 - (C3*C6)
    A6 = C1*C3 - (C2*C4)

    # Calculo de las constantes K
    K1 = A2*A4 + A3*A6
    K2 = A3*A4 + A5*A6
    K3 = (A1**2-A2**2-A3**2-A4**2-A6**2)/(2)

    # Calculamos beta3
    beta31 = 2*np.degrees(arctan2((K2 + sqrt(K1**2 + K2 **2 - K3**2)),(K1+K3)))
    beta32 = 2*np.degrees(arctan2((K2 - sqrt(K1**2 + K2 **2 - K3**2)),(K1+K3)))

    if (round_well(beta31) == round_well(alpha3)):
        
        beta3 = beta32

    else:
        
        beta3 = beta31

    # Calculamos beta2
    beta21 = np.degrees(arctan2(-((A3*sin(np.radians(beta31)) + A2*cos(np.radians(beta31))) + A4),-(A5*sin(np.radians(beta31))+A3*cos(np.radians(beta31))+A6)))
    beta22 = np.degrees(arctan2(-((A3*sin(np.radians(beta32)) + A2*cos(np.radians(beta32))) + A4),-(A5*sin(np.radians(beta32))+A3*cos(np.radians(beta32))+A6)))

    if (round_well(beta21) == round_well(alpha2)):
        
        beta2 = beta22

    else:
        
        beta2 = beta21

    #########################################################
    # CALCULOS PARA EL LADO DERECHO

    # Vectores radio R4, R5 y R6
    R4 = P1 - O4
    R5 = P2 - O4
    R6 = P3 - O4

    # Magnitudes de R4, R5 y R6
    M_R4 = sqrt(R4[0]**2 + R4[1]**2)
    M_R5 = sqrt(R5[0]**2 + R5[1]**2)
    M_R6 = sqrt(R6[0]**2 + R6[1]**2)

    # Angulos de R4, R5 y R6
    zeta4 = np.degrees(arctan2(R4[1], R4[0])) # Angulo de R1
    zeta5 = np.degrees(arctan2(R5[1], R5[0])) # Angulo de R2
    zeta6 = np.degrees(arctan2(R6[1], R6[0])) # Angulo de R3

    # Angulos a calcular
    gamma2 = 0
    gamma3 = 0

    #########################################################
    # CALCULOS PARA GAMMA2 Y GAMMA3

    # Calculo de las constantes C
    C1 = M_R6*cos(np.radians(alpha2 + zeta6)) - M_R5*cos(np.radians(alpha3 + zeta5))
    C2 = M_R6*sin(np.radians(alpha2 + zeta6)) - M_R5*sin(np.radians(alpha3 + zeta5))
    C3 = M_R4*cos(np.radians(alpha3 + zeta4)) - M_R6*cos(np.radians(zeta6))
    C4 = -M_R4*sin(np.radians(alpha3 + zeta4)) + M_R6*sin(np.radians(zeta6))
    C5 = M_R4*cos(np.radians(alpha2 + zeta4)) - M_R5*cos(np.radians(zeta5))
    C6 = -M_R4*sin(np.radians(alpha2 + zeta4)) + M_R5*sin(np.radians(zeta5))

    # Calculo de las constantes A
    A1 = -(C3**2) - (C4**2)
    A2 = C3*C6 - (C4*C5)
    A3 = -(C4*C6) - (C3*C5)
    A4 = C2*C3 + C1*C4
    A5 = C4*C5 - (C3*C6)
    A6 = C1*C3 - (C2*C4)

    # Calculo de las constantes K
    K1 = A2*A4 + A3*A6
    K2 = A3*A4 + A5*A6
    K3 = (A1**2-A2**2-A3**2-A4**2-A6**2)/(2)

    #print(C1, C2, C3, C4, C5, C6, A1, A2, A3, A4, A5, A6, K1, K2, K3)

    # Calculamos gamma3
    gamma31 = 2*np.degrees(arctan2((K2 + sqrt(K1**2 + K2 **2 - K3**2)),(K1+K3)))
    gamma32 = 2*np.degrees(arctan2((K2 - sqrt(K1**2 + K2 **2 - K3**2)),(K1+K3)))

    if ((round_well(gamma31) == round_well(alpha3)) or ((round_well(gamma31 - 360) == round_well(alpha3)))):
        
        gamma3 = gamma32

    else:
        
        gamma3 = gamma31

    # Calculamos gamma2
    gamma21 = np.degrees(arctan2(-((A3*sin(np.radians(gamma31)) + A2*cos(np.radians(gamma31))) + A4),-(A5*sin(np.radians(gamma31))+A3*cos(np.radians(gamma31))+A6)))
    gamma22 = np.degrees(arctan2(-((A3*sin(np.radians(gamma32)) + A2*cos(np.radians(gamma32))) + A4),-(A5*sin(np.radians(gamma32))+A3*cos(np.radians(gamma32))+A6)))

    if (round_well(gamma21) == round_well(alpha2)):
        
        gamma2 = gamma22

    else:
        
        gamma2 = gamma21

    #########################################################
    # CALCULAMOS LA MATRIZ DEL SISTEMA DE ECUACIONES (LADO IZQUIERDO)

    A = cos(np.radians(beta2)) - 1
    B = sin(np.radians(beta2))
    C = cos(np.radians(alpha2)) - 1
    D = sin(np.radians(alpha2))
    E = M_P21*cos(np.radians(delta2))
    F = cos(np.radians(beta3)) - 1
    G = sin(np.radians(beta3))
    H = cos(np.radians(alpha3)) -1
    K = sin(np.radians(alpha3))
    L = M_P31*cos(np.radians(delta3))
    M = M_P21*sin(np.radians(delta2))
    N = M_P31*sin(np.radians(delta3))

    A = np.matrix([[A,-B,C,-D],[F,-G,H,-K],[B,A,D,C],[G,F,K,H]])
    b = np.matrix([[E],[L],[M],[N]])
    x = (A**-1)*b

    W1 = np.array([x[0,0], x[1,0]])
    Z1 = np.array([x[2,0], x[3,0]])

    #########################################################
    # CALCULAMOS LA MATRIZ DEL SISTEMA DE ECUACIONES (LADO DERECHO)

    A = cos(np.radians(gamma2)) - 1
    B = sin(np.radians(gamma2))
    C = cos(np.radians(alpha2)) - 1
    D = sin(np.radians(alpha2))
    E = M_P21*cos(np.radians(delta2))
    F = cos(np.radians(gamma3)) - 1
    G = sin(np.radians(gamma3))
    H = cos(np.radians(alpha3)) -1
    K = sin(np.radians(alpha3))
    L = M_P31*cos(np.radians(delta3))
    M = M_P21*sin(np.radians(delta2))
    N = M_P31*sin(np.radians(delta3))

    A = np.matrix([[A,-B,C,-D],[F,-G,H,-K],[B,A,D,C],[G,F,K,H]])
    b = np.matrix([[E],[L],[M],[N]])
    x = (A**-1)*b

    U1 = np.array([x[0,0], x[1,0]])
    S1 = np.array([x[2,0], x[3,0]])

    #########################################################
    # Salidas
    G1 = O4 - O2
    V1 = O4 + U1 - (O2 + W1)
    auxiliar = angle(V1, Z1)

    L1 = sqrt(G1[0]**2 + G1[1]**2)
    L2 = sqrt(W1[0]**2 + W1[1]**2)
    L3 = sqrt(U1[0]**2 + U1[1]**2)
    L4 = sqrt(V1[0]**2 + V1[1]**2)
    L5 = sqrt(S1[0]**2 + S1[1]**2)
    AP = sqrt(Z1[0]**2 + Z1[1]**2)

    theta = np.degrees(arctan2(W1[1], W1[0])) # Angulo de W1

    if theta < 0:
        theta = theta + 360
    
    O_2 = O2.tolist()
    O_4 = O4.tolist()
    P1 = [O2x,-O2y,7]
    P2 = [O4x,-O4y,7]
    P6 = [O2x,-O2y,-7]
    P7 = [O4x,-O4y,-7]
    O_2 = [O2x,O2y]
    O_4 = [P2[0]-O_2[0],-P2[1]-O_2[1]]

    if ache == 0:
        return P1,P2,P6,P7
    elif ache == 1:
        return O_2,O_4,L1,L2,L3,L4,L5,auxiliar,theta
    else:
        return beta2, beta3, gamma2, gamma3, L1,L2,L3,L4,L5,AP

def calcular():
    # Asociar las entradas a las variables
    global P1x, P1y, P2x, P2y, P3x, P3y, O2x, O2y, O4x, O4y, alpha2, alpha3
    P1x = float(entry_P1x.get())
    P1y = float(entry_P1y.get())
    P2x = float(entry_P2x.get())
    P2y = float(entry_P2y.get())
    P3x = float(entry_P3x.get())
    P3y = float(entry_P3y.get())
    O2x = float(entry_O2x.get())
    O2y = float(entry_O2y.get())
    O4x = float(entry_O4x.get())
    O4y = float(entry_O4y.get())
    alpha2 = float(entry_alpha2.get())
    alpha3 = float(entry_alpha3.get())
    
    # Obtener los valores usando la función entradas
    beta2, beta3, gamma2, gamma3, L1, L2, L3, L4, L5, AP = entradas(3)

    # Crear una nueva ventana
    ventana_resultados = tk.Toplevel(root)
    ventana_resultados.title("Resultados")

    # Mostrar los valores en la nueva ventana
    tk.Label(ventana_resultados, text="beta2: ").grid(row=0, column=0, sticky="e")
    tk.Label(ventana_resultados, text=beta2).grid(row=0, column=1, sticky="w")

    tk.Label(ventana_resultados, text="beta3: ").grid(row=1, column=0, sticky="e")
    tk.Label(ventana_resultados, text=beta3).grid(row=1, column=1, sticky="w")

    tk.Label(ventana_resultados, text="gamma2: ").grid(row=2, column=0, sticky="e")
    tk.Label(ventana_resultados, text=gamma2).grid(row=2, column=1, sticky="w")

    tk.Label(ventana_resultados, text="gamma3: ").grid(row=3, column=0, sticky="e")
    tk.Label(ventana_resultados, text=gamma3).grid(row=3, column=1, sticky="w")

    tk.Label(ventana_resultados, text="Longitudes de eslabones:").grid(row=4, column=0, columnspan=2, pady=5)

    tk.Label(ventana_resultados, text="Longitud entre centros: ").grid(row=5, column=0, sticky="e")
    tk.Label(ventana_resultados, text=L1).grid(row=5, column=1, sticky="w")

    tk.Label(ventana_resultados, text="Longitud de O2 a A: ").grid(row=6, column=0, sticky="e")
    tk.Label(ventana_resultados, text=L2).grid(row=6, column=1, sticky="w")

    tk.Label(ventana_resultados, text="Longitud de O4 a B: ").grid(row=7, column=0, sticky="e")
    tk.Label(ventana_resultados, text=L3).grid(row=7, column=1, sticky="w")

    tk.Label(ventana_resultados, text="Longitud entre A y B: ").grid(row=8, column=0, sticky="e")
    tk.Label(ventana_resultados, text=L4).grid(row=8, column=1, sticky="w")

    tk.Label(ventana_resultados, text="Longitud entre B y P: ").grid(row=9, column=0, sticky="e")
    tk.Label(ventana_resultados, text=L5).grid(row=9, column=1, sticky="w")

    tk.Label(ventana_resultados, text="Longitud entre A y P: ").grid(row=10, column=0, sticky="e")
    tk.Label(ventana_resultados, text=AP).grid(row=10, column=1, sticky="w")

# Crear la ventana principal
root = tk.Tk()
root.title("Configuración")

# Crear y colocar las Entry y Labels
entry_P1x = tk.Entry(root)
entry_P1x.grid(row=0, column=1, padx=5, pady=5)
label_P1x = tk.Label(root, text="P1x")
label_P1x.grid(row=0, column=0, padx=5, pady=5)

entry_P1y = tk.Entry(root)
entry_P1y.grid(row=1, column=1, padx=5, pady=5)
label_P1y = tk.Label(root, text="P1y")
label_P1y.grid(row=1, column=0, padx=5, pady=5)

entry_P2x = tk.Entry(root)
entry_P2x.grid(row=2, column=1, padx=5, pady=5)
label_P2x = tk.Label(root, text="P2x")
label_P2x.grid(row=2, column=0, padx=5, pady=5)

entry_P2y = tk.Entry(root)
entry_P2y.grid(row=3, column=1, padx=5, pady=5)
label_P2y = tk.Label(root, text="P2y")
label_P2y.grid(row=3, column=0, padx=5, pady=5)

entry_P3x = tk.Entry(root)
entry_P3x.grid(row=4, column=1, padx=5, pady=5)
label_P3x = tk.Label(root, text="P3x")
label_P3x.grid(row=4, column=0, padx=5, pady=5)

entry_P3y = tk.Entry(root)
entry_P3y.grid(row=5, column=1, padx=5, pady=5)
label_P3y = tk.Label(root, text="P3y")
label_P3y.grid(row=5, column=0, padx=5, pady=5)

entry_O2x = tk.Entry(root)
entry_O2x.grid(row=6, column=1, padx=5, pady=5)
label_O2x = tk.Label(root, text="O2x")
label_O2x.grid(row=6, column=0, padx=5, pady=5)

entry_O2y = tk.Entry(root)
entry_O2y.grid(row=7, column=1, padx=5, pady=5)
label_O2y = tk.Label(root, text="O2y")
label_O2y.grid(row=7, column=0, padx=5, pady=5)

entry_O4x = tk.Entry(root)
entry_O4x.grid(row=8, column=1, padx=5, pady=5)
label_O4x = tk.Label(root, text="O4x")
label_O4x.grid(row=8, column=0, padx=5, pady=5)

entry_O4y = tk.Entry(root)
entry_O4y.grid(row=9, column=1, padx=5, pady=5)
label_O4y = tk.Label(root, text="O4y")
label_O4y.grid(row=9, column=0, padx=5, pady=5)

entry_alpha2 = tk.Entry(root)
entry_alpha2.grid(row=10, column=1, padx=5, pady=5)
label_alpha2 = tk.Label(root, text="alpha2")
label_alpha2.grid(row=10, column=0, padx=5, pady=5)

entry_alpha3 = tk.Entry(root)
entry_alpha3.grid(row=11, column=1, padx=5, pady=5)
label_alpha3 = tk.Label(root, text="alpha3")
label_alpha3.grid(row=11, column=0, padx=5, pady=5)

# Crear y colocar el botón de calcular
boton_calcular = tk.Button(root, text="Calcular", command=calcular)
boton_calcular.grid(row=12, column=0, columnspan=2, pady=10)

# Iniciar el bucle principal
root.mainloop()

# Configuración de ventana

WIDTH,HEIGHT = 800,600
pygame.display.set_caption("No queríamos que le pongas 0 a los demás, pero no teníamos alternativa.")
screen = pygame.display.set_mode((WIDTH,HEIGHT))

# Variables

scale = 10
circle_pos = [WIDTH/2,2*HEIGHT/3]
angle = 0
angle1 = pi/12
help = 0
help2 = 0
points = []
ene = 0

# Puntos

P1,P2,P6,P7 = entradas(0)

def calc(plano):
    
    O_2,O_4,L1,L2,L3,L4,L5,auxiliar,theta = entradas(1)

    rota = arctan2(O_4[1],O_4[0])
    AP = sqrt(L5**2 + L3**2 - 2*L3*L5*cos(auxiliar))
    O_4_rot = [L1, 0]
    rotation = np.matrix([
            [cos(-rota), -sin(-rota), 0],
            [sin(-rota), cos(-rota), 0],
            [0, 0, 1],
        ])

    K1 = L1/L2
    K2 = L1/L3
    K3 = (L2**2 - L4**2 + L3**2 + L1**2)/(2*L2*L3)

    P3 = []
    P4 = []
    P5 = []

    for angle in range(0, -268, -1):
        
        angle_rad = angle*pi/360
        mu = theta*pi/180 - rota + angle_rad

        A = [L2 * cos(mu), L2 * sin(mu),plano]

        R = cos(mu) - K1 - K2*cos(mu) + K3
        S = -2*sin(mu)
        T = K1 - (K2 + 1)*cos(mu) + K3

        alpha_1 = 2*arctan2((-S - sqrt(S**2 - 4*R*T)), (2*R))
    
        B = [O_4_rot[0] + L3*cos(alpha_1), L3*sin(alpha_1),plano]
        
        delta = arctan2((B[1] - A[1]),(B[0] - A[0]))
    
        omega = delta + auxiliar
        
        P = [A[0] + AP*cos(omega), A[1] + AP*sin(omega), plano]

        A1 = [L2 * cos(mu), - L2 * sin(mu),plano]
        B1 = [O_4_rot[0] + L3*cos(alpha_1), -L3*sin(alpha_1),plano]
        P1 = [A[0] + AP*cos(omega), -(A[1] + AP*sin(omega)), plano]

        A = np.dot(rotation, A1)
        B = np.dot(rotation, B1)
        P = np.dot(rotation, P1)

        A = [A[0, 0] + O_2[0], A[0, 1] - O_2[1], A[0, 2]]
        B = [B[0, 0] + O_2[0], B[0, 1] - O_2[1], B[0, 2]]
        P = [P[0, 0] + O_2[0], P[0, 1] - O_2[1], P[0, 2]]

        P3.append(A)
        P4.append(B)
        P5.append(P)
    
    return P3,P4,P5

P3,P4,P5 = calc(7)
P8,P9,P10 = calc(-7)

def get_coords(i):
    points = [np.matrix([[0, 0, 7]]), np.matrix([[25,  0,  7]]), np.matrix([[ 25, -14,   7]]),
            np.matrix([[  0, -14,   7]]), np.matrix([[ 0,  0, -7]]), np.matrix([[25,  0, -7]]),
            np.matrix([[ 25, -14,  -7]]), np.matrix([[  0, -14,  -7]]), np.matrix([P1]),
            np.matrix([P2]), np.matrix([P3[i]]), np.matrix([P4[i]]),
            np.matrix([P5[i]]), np.matrix([P6]), np.matrix([P7]),
            np.matrix([P8[i]]), np.matrix([P9[i]]), np.matrix([P10[i]])]
    return points

points = get_coords(ene)

projection_matrix = np.matrix([
    [1,0,0],
    [0,1,0],
    [0,0,0]
])

projected_points = [
    [n, n] for n in range(len(points))
]

def connect_points(i, j, points):
    pygame.draw.line(
        screen, BLACK, (points[i][0], points[i][1]), (points[j][0], points[j][1]))

# Empezamos B)

clock = pygame.time.Clock()

while True:

    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            if event.key == pygame.K_RIGHT:
                if help == 1:
                    help = 0
                else:
                    help = -1
            if event.key == pygame.K_LEFT:
                if help == -1:
                    help = 0
                else:
                    help = 1
            if event.key == pygame.K_DOWN:
                if help2 == 1:
                    help2 = 0
                else:
                    help2 = -1
            if event.key == pygame.K_UP:
                if help2 == -1:
                    help2 = 0
                else:
                    help2 = 1
    
    # Sección de actualización

    rotation_z = np.matrix([
        [cos(angle1), -sin(angle1), 0],
        [sin(angle1), cos(angle1), 0],
        [0, 0, 1],
    ])

    rotation_y = np.matrix([
        [cos(angle), 0, sin(angle)],
        [0, 1, 0],
        [-sin(angle), 0, cos(angle)],
    ])

    rotation_x = np.matrix([
        [1, 0, 0],
        [0, cos(angle1), -sin(angle1)],
        [0, sin(angle1), cos(angle1)],
    ])

    if help == 1:
        angle += 0.01
    elif help == -1:
        angle -= 0.01
    
    if help2 == 1:
        ene += 1
        if ene == 268:
            ene = 267
            help2 == 0
        points = get_coords(ene)
    elif help2 == -1:
        ene -= 1
        if ene == -1:
            ene = 0
            help2 == 0
        points = get_coords(ene)
        
    screen.fill(WHITE)
    
    # Sección de dibujo

    i = 0
    for point in points:
        rotated2d = np.dot(rotation_y, point.reshape((3,1)))
        #rotated2d = np.dot(rotation_z, rotated2d)
        rotated2d = np.dot(rotation_x, rotated2d)
        
        projected2d = np.dot(projection_matrix, rotated2d)
        
        x = int(projected2d[0][0]*scale) + circle_pos[0]
        y = int(projected2d[1][0]*scale) + circle_pos[1]

        projected_points[i] = [x,y]
        
        pygame.draw.circle(screen, BLACK, (x,y), 5)
        i += 1
    
    for p in range(4):
        connect_points(p, (p+1) % 4, projected_points)
        connect_points(p+4, ((p+1) % 4) + 4, projected_points)
        connect_points(p, (p+4), projected_points)
    
    connect_points(9, 11, projected_points)
    connect_points(10, 12, projected_points)
    connect_points(11, 12, projected_points)
    connect_points(10, 11, projected_points)
    connect_points(10, 8, projected_points)

    connect_points(13, 15, projected_points)
    connect_points(14, 16, projected_points)
    connect_points(15, 16, projected_points)
    connect_points(15, 17, projected_points)
    connect_points(16, 17, projected_points)

    connect_points(17, 12, projected_points)
    connect_points(15, 10, projected_points)
    connect_points(16, 11, projected_points)
    

    pygame.display.update()

