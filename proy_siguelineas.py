'''
Proyecto
'''

import copy
import numpy as np
import random
import vrep
import sys
import time
import math
import matplotlib.pyplot as plt

class Individuo:
    def __init__(self, solucion):
        self.solucion = solucion

class Problema:
    MIN_VALUE = 0.02
    MAX_VALUE = 1.0
    def __init__(self, clientID):
        self.clientID = clientID

    def fitness(self, vector):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot) # Connect to V-REP
        if self.clientID == -1:
            sys.exit('connection failed')
        else:
            robot = Robot(self.clientID, vector[0], vector[1])
            f = robot.run()
            vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
            time.sleep(2)
        return f

class DE:
    def __init__(self, cantidad_individuos, dimensiones, F, c, problema, generaciones):
        self.cantidad_individuos = cantidad_individuos
        self.dimensiones = dimensiones
        self.F = F
        self.c = c
        self.problema = problema
        self.generaciones = generaciones
        self.individuos = []
        self.mejor_historico = np.inf
        self.rango = self.problema.MAX_VALUE - self.problema.MIN_VALUE
        self.mejor = np.inf
        self.u = []
        self.mejor_individuo = 0
        self.mejor_fitness = 0
        self.fitness_x = []

    def crearIndividuos(self):
        for i in range(self.cantidad_individuos):
            solucion = np.random.random(size = self.dimensiones) * self.rango + self.problema.MIN_VALUE
            individuo = Individuo(solucion)
            self.individuos.append(individuo)

    def run(self):
        self.crearIndividuos()
        self.mejor = copy.deepcopy(self.individuos[0])
        self.mejor_fitness = self.problema.fitness(self.mejor.solucion)
        self.mejor_individuo = copy.deepcopy(self.individuos[0].solucion)
        for i in range(len(self.individuos)):
            self.fitness_x.append(self.problema.fitness(self.individuos[i].solucion))
            if self.fitness_x[i] < self.mejor_fitness:
                self.mejor_individuo = copy.deepcopy(self.individuos[i].solucion)
                self.mejor_fitness = self.fitness_x[i]
        generacion = 0
        ui = []
        puntos = []
        promedios = []
        font = {'color' : 'darkblue','size' : 10}
        plt.figure(num='Simulation', figsize=(480/118, 480/118), dpi=118)
        plt.ion()
        #plt.show()
        while generacion < self.generaciones:
            self.u = []
            for i in range(len(self.individuos)):
                idxs = random.sample(range(0, self.cantidad_individuos), 3)
                while i in idxs:
                    idxs = random.sample(range(0, self.cantidad_individuos), 3)
                vi = np.abs(self.individuos[idxs[0]].solucion + (self.F * np.subtract(self.individuos[idxs[1]].solucion, self.individuos[idxs[2]].solucion)))###
                Jr = random.randint(0, self.dimensiones)
                ui = []
                for j in range(self.dimensiones):
                    rcj = random.random()
                    if rcj < self.c or j == Jr:
                        ui.append(vi[j])
                    else:
                        ui.append(self.individuos[i].solucion[j])
                self.u.append(copy.deepcopy(ui))
            for i in range(len(self.individuos)):
                fitness_ui = self.problema.fitness(self.u[i])
                fitness_xi = self.fitness_x[i]
                if fitness_ui < fitness_xi:
                    self.individuos[i].solucion = copy.deepcopy(self.u[i])
                    self.fitness_x[i] = fitness_ui
                    if fitness_ui < self.mejor_fitness:
                        self.mejor_individuo = copy.deepcopy(self.u[i])
                        self.mejor_fitness = fitness_ui

            puntos.append(generacion)
            promedios.append(self.mejor_fitness)
            line1, = plt.plot(puntos, promedios, linewidth=4, color='#0000ff')
            plt.title('Time over generations')
            plt.xlabel('Generation', fontdict=font)
            plt.ylabel('Time', fontdict=font)
            plt.draw()
            plt.pause(0.001)
            print('Generación:', generacion, 'Mejor:', self.mejor_individuo, ':', self.mejor_fitness)
            generacion += 1

class Robot:
    def __init__(self, clientID, nominalLinearVelocity, adjust):
        self.clientID = clientID
        self.left_joint_handle = 0
        self.right_joint_handle = 0
        self.left_sensor_handle = 0
        self.middle_sensor_handle = 0
        self.right_sensor_handle = 0
        self.nominalLinearVelocity = nominalLinearVelocity
        self.wheelRadius=0.027
        self.interWheelDistance=0.119
        self.s = 1
        self.adjust = adjust
        self.handle = 0
        self.position = []
        self.initial_position = []
        self.away = False
        self.fitness = math.inf

        self.initSensors()
        self.initJoints()
        error_code, self.handle = vrep.simxGetObjectHandle(self.clientID, 'LineTracer', vrep.simx_opmode_oneshot_wait)
        return_code, self.initial_position = vrep.simxGetObjectPosition(self.clientID, self.handle, -1, vrep.simx_opmode_streaming)

    def setNominalLinearVelocity(self, nominalLinearVelocity):
        self.nominalLinearVelocity = nominalLinearVelocity

    def setAdjust(self, adjust):
        self.adjust = adjust


    def initSensors(self):
        error_code, self.left_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'LeftSensor', vrep.simx_opmode_oneshot_wait)
        return_code, left_state, left_packets = vrep.simxReadVisionSensor(self.clientID, self.left_sensor_handle, vrep.simx_opmode_streaming)

        error_code, self.middle_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'MiddleSensor', vrep.simx_opmode_oneshot_wait)
        return_code, middle_state, middle_packets = vrep.simxReadVisionSensor(self.clientID, self.middle_sensor_handle, vrep.simx_opmode_streaming)

        error_code, self.right_sensor_handle = vrep.simxGetObjectHandle(self.clientID, 'RightSensor', vrep.simx_opmode_oneshot_wait)
        return_code, right_state, right_packets = vrep.simxReadVisionSensor(self.clientID, self.right_sensor_handle, vrep.simx_opmode_streaming)

    def initJoints(self):
        error_code, self.left_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
        error_code, self.right_joint_handle = vrep.simxGetObjectHandle(self.clientID, 'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)

    def run(self):
        self.away = False
        while self.initial_position == [0.0, 0.0, 0.0]:
            return_code, self.initial_position = vrep.simxGetObjectPosition(self.clientID, self.handle, -1, vrep.simx_opmode_buffer)
            t = time.time()
        return_code, self.initial_position = vrep.simxGetObjectPosition(self.clientID, self.handle, -1, vrep.simx_opmode_buffer)
        t = time.time()
        while True:
            return_code, left_state, left_packets = vrep.simxReadVisionSensor(self.clientID, self.left_sensor_handle, vrep.simx_opmode_buffer)
            return_code, middle_state, middle_packets = vrep.simxReadVisionSensor(self.clientID, self.middle_sensor_handle, vrep.simx_opmode_buffer)
            return_code, right_state, right_packets = vrep.simxReadVisionSensor(self.clientID, self.right_sensor_handle, vrep.simx_opmode_buffer)

            linearVelocityLeft=self.nominalLinearVelocity*self.s
            linearVelocityRight=self.nominalLinearVelocity*self.s

            if left_state:
                linearVelocityRight=linearVelocityRight*self.adjust
            if right_state:
                linearVelocityLeft=linearVelocityLeft*self.adjust

            if left_state and right_state and self.away:
                self.fitness = math.inf
                break

            vrep.simxSetJointTargetVelocity(self.clientID, self.left_joint_handle, linearVelocityLeft/(self.s*self.wheelRadius), vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(self.clientID, self.right_joint_handle, linearVelocityRight/(self.s*self.wheelRadius), vrep.simx_opmode_streaming)

            return_code, self.position = vrep.simxGetObjectPosition(self.clientID, self.handle, -1, vrep.simx_opmode_buffer)
            euclidean_dist = (((self.position[0] - self.initial_position[0]) ** 2) + ((self.position[1] - self.initial_position[1]) ** 2)) ** (1/2)
            if euclidean_dist > 0.05:
                self.away = True
            if euclidean_dist < 0.05 and self.away:
                self.away = False
                self.fitness = time.time() - t
                t = time.time()
                break
            time.sleep(0.01)
        return self.fitness


def main():
    time.sleep(10)
    vrep.simxFinish(-1) # just in case, close all opened connections
    time.sleep(2)
    clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to V-REP
    problema = Problema(clientID)
    de = DE(8, 2, 0.65, 0.4, problema, 30)
    de.run()


if  __name__ == '__main__':
    main()
