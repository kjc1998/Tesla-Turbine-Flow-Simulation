import pandas as pd
from enum import Enum
import numpy as np
import scipy
from numpy import sqrt, sin, cos, pi
from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# variable definition falls under constant(const), derived(drv), variable(var)
const_density = 997
const_kinematicViscosity = 8.917*10**-7
const_dynamicViscosity = const_kinematicViscosity * const_density


class flow_parameters():
    def alternator(self, omega, torque):
        self.var_omega = omega
        self.var_torque = torque

    def disc(self, inner_radius, outer_radius, disc_thickness, disc_spacing, Fpo=1):
        self.var_innerRadius = inner_radius
        self.var_outerRadius = outer_radius
        self.var_discThickness = disc_thickness
        self.var_discSpacing = disc_spacing
        self.var_Fpo = Fpo

    def inlet(self, volume_flow_rate, nozzle_angle):
        self.var_volumeFlowRate = volume_flow_rate
        self.var_nozzleAangle = nozzle_angle  # 0 being parallel to disc

    def derived(self):
        self.drv_massFlowRate = self.var_volumeFlowRate*const_density  # not confirmed yet
        self.drv_velocityNet = self.var_volumeFlowRate / \
            (2*pi*self.var_outerRadius*self.var_discSpacing)
        self.drv_velocityRadial = self.drv_velocityNet * \
            sin(self.var_nozzleAangle)
        self.drv_velocityTangential = self.drv_velocityNet * \
            cos(self.var_nozzleAangle)

        self.drv_outerDiscSpeed = self.var_omega*self.var_outerRadius
        self.drv_innerDiscSpeed = self.var_omega*self.var_innerRadius
        self.drv_aspectRatio = self.var_discSpacing/self.var_outerRadius

        self.drv_velocityRadialRatio = self.drv_velocityRadial/self.drv_outerDiscSpeed
        self.drv_velocityTangentialRatio = self.drv_velocityTangential/self.drv_outerDiscSpeed
        if(self.drv_velocityTangentialRatio < 1):
            raise Exception(
                "The inlet's tangential velocity is lower than tip disc's speed.")

        self.drv_nProfile = 3*self.var_Fpo - 1
        self.drv_po = 24*self.var_Fpo

        self.drv_rotationalRe = (
            self.var_omega*self.var_discSpacing**2)/const_kinematicViscosity
        self.drv_discRe = self.drv_rotationalRe/self.drv_aspectRatio
        self.drv_nozzleRe = 2*pi*self.drv_discRe*self.drv_velocityRadialRatio
        self.drv_Rem = 4*self.drv_rotationalRe*self.drv_velocityRadialRatio


def z_profile(instance, zs):
    return ((instance.drv_nProfile+1)/instance.drv_nProfile)*(1-(2*zs/instance.var_discSpacing)**instance.drv_nProfile)


def tangential_ODE(y, rs, instance):
    first_term = (-2*instance.drv_nProfile+1)/(instance.drv_nProfile + 1)
    second_term = -y/rs
    third_term = 8*(2*instance.drv_nProfile+1)*rs*y/instance.drv_Rem
    return first_term+second_term+third_term


def pressure_ODE(y, rs, instance, first_answer_dic):
    first_term = ((4*instance.drv_nProfile + 1) * instance.drv_velocityRadialRatio**2)\
        / ((2*instance.drv_nProfile + 1)*rs**3)
    second_term = tangential_ODE(interpolation(rs, first_answer_dic), rs, instance)\
        * rs**2
    thrid_term = 4*interpolation(rs, first_answer_dic)
    fourth_term = 2*rs
    fifth_term = 32*(instance.drv_nProfile - 1) * \
        (instance.drv_velocityRadialRatio**2)/(rs*instance.drv_Rem)
    return first_term + second_term + thrid_term + fourth_term + fifth_term


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


def solver(rs, instance, const_firstODEBoundary, const_secondODEBoundary):
    first_solution = odeint(
        tangential_ODE, const_firstODEBoundary, rs, args=(instance,))
    zip_iterator = zip(rs, first_solution)
    first_solutionDictionary = dict(zip_iterator)

    second_solution = odeint(
        pressure_ODE, const_secondODEBoundary, rs, args=(
            instance, first_solutionDictionary)
    )
    return first_solution, second_solution


def power_theoretical(rs, instance, first_solution):
    rs1 = rs*instance.var_outerRadius
    first_solution1 = first_solution*instance.drv_outerDiscSpeed
    mean_ratio = (1/instance.drv_nProfile)*2**instance.drv_nProfile
    mean_firstSolution = abs(mean_ratio*first_solution1)
    tau_wall = (12*instance.var_Fpo*const_dynamicViscosity *
                mean_firstSolution)/(2*instance.var_discSpacing)
    integral_term = np.flip(np.transpose(
        [np.power(rs1, 2)])*np.array(tau_wall)).ravel()
    rs1 = np.flip(rs1)
    power = 2*pi*instance.var_omega*np.trapz(integral_term, x=rs1)
    return power


def mechanical_efficiency(instance):
    innnerOuterRatio = instance.var_innerRadius/instance.var_outerRadius
    return 1 - (instance.drv_velocityTangentialRatio - 1 + innnerOuterRatio)*innnerOuterRatio/(instance.drv_velocityTangentialRatio)


def theoretical_efficiency(instance, first_answer, second_answer):
    '''SFEE: P/density + 1/2 * v^2 = Constant'''
    first_term = abs(second_answer[-1][0])

    innnerOuterRatio = instance.var_innerRadius/instance.var_outerRadius
    mean_ratio = (1/instance.drv_nProfile)*2**instance.drv_nProfile
    exit_tangential = first_answer[-1][0]*instance.drv_outerDiscSpeed

    mean_exitTangential = abs(mean_ratio*exit_tangential)
    mean_exitRadial = (1/innnerOuterRatio) * \
        instance.drv_velocityRadial*mean_ratio

    mean_exitVelocityMagnitude = (mean_exitRadial**2 + mean_exitTangential**2)
    mean_inletVelocityMagnitude = (instance.drv_velocityNet)**2

    second_term = 0.5*(mean_inletVelocityMagnitude -
                       mean_exitVelocityMagnitude)/(instance.drv_outerDiscSpeed**2)  # minimal impact
    denominator = first_term + second_term
    numerator = power_theoretical(
        rs, instance, first_answer)/instance.drv_outerDiscSpeed**2
    return numerator/denominator


'''
# instance definition
KJ = flow_parameters()
KJ.alternator(50, 1000)
KJ.disc(0.05, 0.25, 0.0002, 0.000465, 1)
KJ.inlet(0.0118, 0.063)
KJ.derived()
# end
print(KJ.drv_nProfile)
print("U0 is: ", KJ.drv_velocityRadialRatio)
print("V0 is: ", KJ.drv_velocityTangentialRatio)
print("Rec is: ", KJ.drv_rotationalRe)
print(
    f"Disc outer radius is: {KJ.var_outerRadius}, inner radius is: {KJ.var_innerRadius}")
'''
'''Solving both ODE sets'''

fig, ax = plt.subplots()
for i in range(4):

    Fpo = (2*(i+1) + 1)/3
    KJ = flow_parameters()
    KJ.alternator(50, 1000)
    KJ.disc(0.05, 0.25, 0.0002, 0.000465, Fpo)
    KJ.inlet(0.0118, 0.063)
    KJ.derived()

    const_firstODEBoundary = KJ.drv_velocityTangentialRatio - 1
    const_secondODEBoundary = 0

    rs = np.linspace(1, KJ.var_innerRadius/KJ.var_outerRadius, 50)
    zs = np.linspace(-KJ.var_discSpacing/2, KJ.var_discSpacing/2, 25)

    first_solution, second_solution = solver(
        rs, KJ, const_firstODEBoundary, const_secondODEBoundary)
    output = power_theoretical(rs, KJ, first_solution)
    rotor_eff = mechanical_efficiency(KJ)
    ax.plot(np.flip(rs), np.flip(second_solution), label=f"{2*(i+1)}")

    tangential_zPoints = np.transpose([np.array(first_solution).ravel()]) * \
        np.array([z_profile(KJ, zs)]) + \
        np.transpose(np.repeat(np.array([rs]), 25, axis=0))
    tangential_relativeZPoints = np.transpose([np.array(first_solution).ravel()]) * \
        np.array([z_profile(KJ, zs)])

    X, Y = np.meshgrid(zs, rs)
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    ax1.set_xlabel('z')
    ax1.set_ylabel('R ratio')
    ax1.set_zlabel('Magnitude')
    surf = ax1.plot_surface(X, Y, tangential_zPoints)
    surf1 = ax1.plot_surface(X, Y, tangential_relativeZPoints)
ax.legend()
plt.show()
