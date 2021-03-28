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


const_density = 997
const_kinematicViscosity = 8.917*10**-7
const_dynamicViscosity = const_kinematicViscosity * const_density

'''
Air:
const_density = 1.184
const_kinematicViscosity = 18.37*10**-6
const_dynamicViscosity = const_kinematicViscosity * const_density
'''


class flow_parameters():
    '''Storing variables'''

    def alternator(self, omega, torque):
        self.var_omega = omega
        self.var_torque = torque

    def disc(self, inner_radius, outer_radius, disc_thickness, disc_spacing):
        self.var_innerRadius = inner_radius
        self.var_outerRadius = outer_radius
        self.var_discThickness = disc_thickness
        self.var_discSpacing = disc_spacing

    def inlet(self, Vr, Vtheta):
        self.inletVradial = Vr
        self.inletVtheta = Vtheta  # 0 being parallel to disc

    def derived(self):
        self.nozzleAngle = np.arctan(abs(self.inletVtheta/self.inletVradial))

        self.volumeFlowRate = 2*pi*self.var_outerRadius * \
            self.var_discSpacing*abs(self.inletVradial)
        self.massFlowRate = self.volumeFlowRate*const_density

        self.innerOuterRatio = self.var_innerRadius/self.var_outerRadius
        self.phi2 = self.inletVradial / \
            (self.var_omega*self.var_outerRadius)
        self.gamma = self.inletVtheta / \
            (self.var_omega*self.var_outerRadius)


def zProfile(zs, instance):
    '''Velocity profile in z axis'''
    return np.array([6*(zs)*(1-zs)]).transpose()


def velocityRadial3D(rs, zs, instance):
    '''3d plot points for radial velocity'''
    return np.array(instance.inletVradial*(1/rs)*zProfile(zs, instance))


def velocitytangential3D(rs, zs, instance, firstAnswer):
    '''3d plot points for tangential velocity'''
    return np.array(instance.inletVtheta*firstAnswer*zProfile(zs, instance))


def bothODE(Y, R, instance):
    '''Coupled ODE equations'''
    v = const_kinematicViscosity
    phi = instance.phi2
    gamma = instance.gamma
    omegaspace = instance.var_omega*instance.var_discSpacing**2
    Y01 = Y[0]

    first_term1 = -(1/R + 10*(v/omegaspace)*(R/phi))*Y01 - 10/(6*(gamma-1))

    second_term1 = R + 2*(gamma - 1)*Y01 + (6/5)*(Y01**2/R)*(gamma - 1)**2
    second_term2 = (6/5)*(phi**2/R**3)-12*(v / omegaspace)*(phi/R)

    return [first_term1, second_term1+second_term2]


def workDone(instance, rs, firstAnswer):
    '''Calculation of work done'''
    firstAnswer1 = np.flip(firstAnswer)
    rs1 = np.flip(rs)

    coef = (12*pi*const_dynamicViscosity*abs(instance.inletVtheta)
            * instance.var_outerRadius**3)/instance.var_discSpacing
    toBeIntegrated = np.power(rs1, 2)*firstAnswer1
    return 2*coef*np.trapz(toBeIntegrated, x=rs1)*instance.var_omega


def efficiency(instance, rs, answer):
    MFR = instance.massFlowRate
    inlet = (np.linalg.norm([instance.inletVradial, instance.inletVtheta]))**2
    VTLast, PLast = abs(answer[-1, 0]), abs(answer[-1, 1])
    VRLast = instance.inletVradial * \
        (instance.var_outerRadius/instance.var_innerRadius)
    outlet = (np.linalg.norm([VTLast, VRLast]))**2
    energyChange = PLast/const_density + 0.5*(inlet-outlet)
    return workDone(instance, rs, answer[:, 0])/energyChange


def Report(instance, rs, answer):
    '''Gamma, pressure difference, and power out'''
    print("Gamma is: ", instance.gamma)
    print("Pressure difference: ", (1/100000)*abs(
        answer[:, 1][-1])*const_density*(instance.var_omega**2)*(instance.var_outerRadius**2), "bar")
    print("Power out is: ", workDone(instance, rs, answer[:, 0]), "Watt")
    print("Efficiency: ", efficiency(instance, rs, answer)*100, "%")


'''
KJ = flow_parameters()
KJ.alternator(500, 100)  # 1000 rpm
# inner radius and outer radius can be amended, disc spacing as well
KJ.disc(0.125, 0.25, 0.005, 0.005)
KJ.inlet(-2, 19.64)
KJ.derived()
print("Volume flow rate is: ", KJ.volumeFlowRate, "m3/s")
initialCondition = [1, 0]

zs = np.linspace(0, 1, 25)
rs = np.linspace(1, KJ.innerOuterRatio, 50)
X, Y = np.meshgrid(rs, zs)
answer = odeint(
    bothODE, initialCondition, rs, args=(KJ,))
answer1, answer2 = answer[:, 0], answer[:, 1]
Report(KJ, rs, answer)
'''

'''
# plots
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.set_xlabel('R ratio')
ax1.set_ylabel('z')
ax1.set_zlabel('Radial velocity (ms-1)')
surf = ax1.plot_surface(X, Y, velocityRadial3D(rs, zs, KJ))

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
ax2.set_xlabel('R ratio')
ax2.set_ylabel('z')
ax2.set_zlabel('Tangential velocity (ms-1)')
surf = ax2.plot_surface(X, Y, velocitytangential3D(rs, zs, KJ, answer1))
plt.show()
'''
'''
# efficiency graph across Nre and V0
testKJ = flow_parameters()
testKJ.alternator(3000, 100)
testKJ.disc(0.0625, 0.125, 0.005, 0.005)
testKJ.inlet(-0.2, 9.82)
testKJ.derived()

rs = np.linspace(1, testKJ.innerOuterRatio, 50)
discSpacingList = np.linspace(0.002, 0.01, 100)
V0List = (testKJ.var_omega*testKJ.var_outerRadius)*np.linspace(0, 1.5, 50)

z_eff = np.zeros((len(discSpacingList), len(V0List)))

for i in range(len(discSpacingList)):
    for j in range(len(V0List)):
        testKJ.var_discSpacing = discSpacingList[i]
        testKJ.inletVtheta = V0List[j]
        testKJ.derived()
        ANSWER = odeint(bothODE, [1, 0], rs, args=(testKJ,))
        z_eff[i, j] = efficiency(testKJ, rs, ANSWER)

X, Y = np.meshgrid(discSpacingList, np.linspace(0.5, 2, 50))
# (testKJ.var_omega/const_kinematicViscosity)*np.power(discSpacingList,2)
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.set_xlabel('discSpacing')
ax1.set_ylabel('V0')
ax1.set_zlabel('efficiency')
surf = ax1.plot_surface(X, Y, z_eff.transpose())
plt.show()
'''
