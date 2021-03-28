import pandas as pd
import numpy as np
from numpy import sqrt, sin, cos, pi
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# variables definition


class variables():
    wall_velocity = 0
    central_velocity_differential = 0
    density = 997
    kinematic_viscosity = 8.917*10**-7

    def __init__(self, inner_radius, outer_radius, disc_spacing):
        '''inner_radius, outer_radius, disc_spacing'''
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.disc_spacing = disc_spacing

    def turbine_parameters(self, torque, omega):
        '''torque, omega'''
        self.angular_velocity = omega
        self.torque_set = torque

    def boundary_conditions(self, Vr_average, Vtheta_average, inlet_pressure):
        '''Vr_average, Vtheta_average, inlet_pressure'''
        self.inlet_velocity = np.array(
            [Vr_average, Vtheta_average])
        self.inlet_pressure = inlet_pressure
        self.wall_velocity = variables.wall_velocity
        self.central_velocity_differential = variables.central_velocity_differential
        self.velocity_magnitude = np.linalg.norm(self.inlet_velocity)

    def dimensionless_number(self):
        self.phi = self.inlet_velocity[0] / \
            (self.angular_velocity*self.outer_radius)
        self.gamma = self.inlet_velocity[1] / \
            (self.angular_velocity*self.outer_radius)


# Equations


def z_velocity_profile(z, instance):
    return 6*(z/instance.disc_spacing)*(1-z/instance.disc_spacing)


def tangential_profile_ODE(y, R, instance):
    v = instance.kinematic_viscosity
    phi = instance.phi
    gamma = instance.gamma
    first_term = -(1/R + 10*(v /
                             (instance.angular_velocity*instance.disc_spacing**2))*(R/phi))*y
    second_term = -10/(6*(gamma-1))
    return first_term+second_term


def radial_velocity_profile(R, z, instance):
    R_profile = np.array(R**-1)
    z_profile = np.array([z_velocity_profile(z, instance)])
    return instance.inlet_velocity[0]*np.transpose(z_profile)*R_profile


def interpolation(key, dictionary):
    try:
        answer = dictionary[key]
    except:
        for keys in dictionary:
            if keys < key:
                upper_key = keys
                break
            else:
                lower_key = keys
                if(list(dictionary.keys()).index(keys) == len(dictionary)-1):
                    answer = dictionary[keys]
                continue
        try:
            answer = ((dictionary[upper_key]-dictionary[lower_key]) /
                      (upper_key-lower_key))*(key-lower_key) + dictionary[lower_key]
        except:
            pass
    return answer


def pressure_drop_ODE(y2, R, instance, first_answer):
    first_term = R + 2*(instance.gamma-1)*interpolation(R, first_answer)
    second_term = (interpolation(R, first_answer)**2/R) * \
        (6/5)*(instance.gamma - 1)**2
    third_term = (6/5)*(instance.phi**2/R**3)
    fourth_term = -12*(instance.kinematic_viscosity /
                       (instance.angular_velocity*instance.disc_spacing**2))*(instance.phi/R)
    return first_term+second_term+third_term+fourth_term


def pressure_drop_ODE_Ori(R, instance, first_answer):
    first_term = R + 2*(instance.gamma-1)*first_answer
    second_term = (first_answer**2/R) * \
        (6/5)*(instance.gamma - 1)**2
    third_term = (6/5)*(instance.phi**2/R**3)
    fourth_term = -12*(instance.kinematic_viscosity /
                       (instance.angular_velocity*instance.disc_spacing**2))*(instance.phi/R)
    return first_term+second_term+third_term+fourth_term


def gradient_pressure_drop(R, Y):
    output = []
    for i in range(len(R)-1):
        output.append((Y[i+1]-Y[i])/(R[i+1]-R[i]))
    return output


# instance
KJ = variables(0.0132, 0.025, 0.005)
KJ.turbine_parameters(50, 1000)
KJ.boundary_conditions(-11.5, 106, 100000)
KJ.dimensionless_number()


# initial conditions
first_ODE_boundary = 1
second_ODE_boundary = 0
rs = np.linspace(1, KJ.inner_radius/KJ.outer_radius, 50)
zs = np.linspace(0, KJ.disc_spacing, 50)


# solving first ODE
tangential_profile_set = odeint(
    tangential_profile_ODE, first_ODE_boundary, rs, args=(KJ,))
print(tangential_profile_set)
tangential_profile_set /= KJ.inlet_velocity[1]
z = z_velocity_profile(zs, KJ)
tangential_profile_disc = KJ.inlet_velocity[1] * \
    np.transpose(z)*tangential_profile_set


# dictionary to map R ratio to first output
zip_iterator = zip(rs, tangential_profile_set)
first_answer_dictionary = dict(zip_iterator)


# solving second ODE, answer in p'
pressure_diff_set = odeint(pressure_drop_ODE, second_ODE_boundary,
                           rs, args=(KJ, first_answer_dictionary))
print(pressure_diff_set)


# solving pressure gradient based on derived PDE equation
pressure_diff_set2 = []
for i in range(len(rs)-1):  # forward scheme method
    pressure_diff_set2.append(
        (pressure_diff_set[i+1]-pressure_diff_set[i])/(rs[i+1]-rs[i]))
np.array(pressure_diff_set2)

# solving pressure gradient based on original equation
pressure_diff_set3 = pressure_drop_ODE_Ori(
    rs.reshape(50, 1), KJ, tangential_profile_set)


# Radial Velocity profile
Radial_Velocity_disc = radial_velocity_profile(rs, zs, KJ)
Radial_Velocity_disc /= -(KJ.inlet_velocity[0])


# plots
X, Y = np.meshgrid(zs, rs)
fig0, ax0 = plt.subplots()
ax0.plot(rs, pressure_diff_set, label="p'_ODE")
ax0.plot(rs[0:-1], pressure_diff_set2, label="dp'/dR_ODE")
ax0.plot(rs, pressure_diff_set3, label="dp'/dR_ori")
ax0.set_xlabel('R ratio')
ax0.set_ylabel('Magnitude')
ax0.legend()

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.set_xlabel('z')
ax1.set_ylabel('R ratio')
ax1.set_zlabel('dimensionless tangential velocity')
surf = ax1.plot_surface(X, Y, tangential_profile_disc)

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_xlabel('z')
ax2.set_ylabel('R ratio')
ax2.set_zlabel('dimensionless radial velocity')
surf = ax2.plot_surface(X, Y, Radial_Velocity_disc.transpose())
plt.show()
