import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import fsolve

class Physical_quantities:
    def __init__(self):
        self.q = 1.6e-19 # заряд в СИ
        self.k = 8.617e-5 # постоянная Больцмана в эВ
        self.h = 4.135e-15 
        self.m = 9.1e-31 # масса электрона в СИ
        self.k_si = 1.38e-23 # постоянная Больцмана в СИ
        self.epsilond_0 = 8.85E-14

class Initial_param:
    def __init__(self):
        self.Eg_300 = 0.67 # эВ
        self.m_dn = 0.56
        self.m_dp = 0.37
        self.Eg_kof = 0.785 # эВ
        self.Eg_alfa = -4.2E-4 # эВ/K
        self.g = 2
        self.epsilond = 16.3
        self.N_E = 1.9E19 #см-3
        self.N_B = 6.6E16 #см-3
        self.N_C = 3.8E15 #см-3
        self.x_E = 69E-4 #см
        self.x_B = 14E-4 #см
        self.x_C = 55E-4 #см
        self.d_E = 2.9E-1 #см
        self.d_B = 10.6E-1 #см
        self.d_C = 6.6E-1 #см
        self.t_n = 1E-6 #c
        self.t_p = 5.6E-7 #c
        self.V_BE = 0.39 #V
        self.V_CB = 9 #V
        self.I_E = 2.6E-3 #A
        self.N_c = 1.04E19 #см-3
        self.N_v = 6E18 #см-3
        self.ni = 2.4E13 #см-3
        self.T = 300 #K
        self.mu_p_E = 1E2 #см2/В/с
        self.mu_n_B = 3.1E3 #см2/В/с
        self.mu_p_C = 1.8E3 #см2/В/с

class Culculation_param():
    def __init__(self):
        Physical_quantities.__init__(self)
        Initial_param.__init__(self)

    def find_fi_EB(self):
        return self.k * self.T * np.log(self.N_E * self.N_B / ((self.ni) ** 2))

    def find_fi_CB(self):
        return self.k * self.T * np.log(self.N_C * self.N_B / self.ni / self.ni)

    def find_Wdep_EB(self):
        return np.sqrt(2*self.epsilond * self.epsilond_0 / self.q * \
            (self.N_E + self.N_B)/(self.N_E * self.N_B) * self.find_fi_EB())

    def find_xdep_E(self):
        return self.N_B / (self.N_B + self.N_E) * self.find_Wdep_EB()

    def find_xdepB1(self):
        return self.find_Wdep_EB() - self.find_xdep_E()

    def find_Wdep_BC(self):
        return np.sqrt(2*self.epsilond * self.epsilond_0 / self.q * \
            (self.N_C + self.N_B)/(self.N_C * self.N_B) * self.find_fi_CB())

    def find_xdep_C(self):
        return self.N_B / (self.N_B + self.N_C) * self.find_Wdep_BC()

    def find_xdepB2(self):
        return self.find_Wdep_BC() - self.find_xdep_C()

    def find_Wdep_EB_r(self):
        return np.sqrt(2*self.epsilond * self.epsilond_0 / self.q * \
            (self.N_E + self.N_B)/(self.N_E * self.N_B) * (self.find_fi_EB() - self.V_BE))

    def find_xdep_E_r(self):
        return self.N_B / (self.N_B + self.N_E) * self.find_Wdep_EB_r()

    def find_xdepB1_r(self):
        return self.find_Wdep_EB_r() - self.find_xdep_E_r()

    def find_Wdep_BC_r(self):
        return np.sqrt(2*self.epsilond * self.epsilond_0 / self.q * \
            (self.N_C + self.N_B)/(self.N_C * self.N_B) * (self.find_fi_CB() + self.V_CB))

    def find_xdep_C_r(self):
        return self.N_B / (self.N_B + self.N_C) * self.find_Wdep_BC_r()

    def find_xdepB2_r(self):
        return self.find_Wdep_BC_r() - self.find_xdep_C_r()

    def find_dF_E(self):
        return self.k * self.T * np.log(self.N_c / self.N_E)
    
    def find_dF_B(self):
        return self.k * self.T * np.log(self.N_v / self.N_B)
   
    def find_dF_C(self):
        return self.k * self.T * np.log(self.N_c / self.N_C)
    
    def find_pE0(self):
        return (self.ni ** 2) / self.N_E
    
    def find_nB0(self):
        return (self.ni ** 2) / self.N_B

    def find_pC0(self):
        return (self.ni ** 2) / self.N_C 

    def find_pE(self):
        return self.find_pE0() * (np.exp(self.V_BE / (self.k * self.T))-1)

    def find_nB1(self):
        return self.find_nB0() * (np.exp(self.V_BE / (self.k * self.T))-1)

    def find_LE(self):
        return np.sqrt(self.k * self.T * self.mu_p_E * self.t_p)

    def find_LB(self):
        return np.sqrt(self.k * self.T * self.mu_n_B * self.t_n)

    def find_LC(self):
        return np.sqrt(self.k * self.T * self.mu_p_C * self.t_p)

    def find_pE_x(self, x):
        return self.find_pE0() + self.find_pE() * np.exp(x / self.find_LE())

    def find_n_B_X(self, x):
        return self.find_nB1()*(1 - ((x - (self.x_E + self.find_xdepB1())) \
            / (self.x_B - self.find_xdepB1() - self.find_xdepB2())))

    def find_p_C_x(self, x):
        return 0 + self.find_pC0() * np.exp(-1 * (x - self.x_E - self.x_B - self.find_xdep_C()) \
            / (self.find_LC()))

    def find_gamma(self):
        tanh = math.tanh((self.x_B - self.find_xdepB1() - self.find_xdepB2()) / (self.find_LB()))
        up = (self.ni ** 2) * self.k * self.T / self.N_E * self.mu_p_E * self.find_LB()
        down = (self.ni ** 2) * self.k * self.T / self.N_B * self.mu_n_B * self.find_LE()
        return (1 + up / down * tanh) ** (-1)

    def find_alfaT(self):
        cosh= math.cosh((self.x_B - self.find_xdepB1() - self.find_xdepB2()) / (self.find_LB()))
        return (cosh) ** (-1)

    def find_Ir0(self):
        return self.q * self.ni * (self.find_xdep_E() + self.find_xdepB1()) / (self.t_n + self.t_p)

    def find_Is0(self):
        tanh = math.tanh((self.x_B - self.find_xdepB1() - self.find_xdepB2()) / (self.find_LB()))
        up = (self.ni ** 2) * self.k * self.T * self.q * self.mu_n_B / self.N_B
        return up / tanh / self.find_LB()

    def find_delta(self):
        return (1 + self.find_Ir0() / self.find_Is0() * np.exp(-self.V_BE / 2 / self.k / self.T)) ** (-1)

    def find_alfa(self):
        return self.find_alfaT() * self.find_delta() * self.find_gamma()

    def find_beta(self):
        return self.find_alfa() / (1 - self.find_alfa())

    def find_Vp(self):
        conc = self.N_B / self.N_C * (self.N_B + self.N_C)
        return self.q / 2 / self.epsilond / self.epsilond_0 * ((self.x_B) ** 2) * conc

    def find_VA(self):
        return self.q / self.epsilond / self.epsilond_0 * ((self.x_B) ** 2) * self.N_B

    def find_IF0(self):
        return np.pi * (self.d_E ** 2) / 4 * self.q * \
            (self.find_pE0() * self.find_LB() / self.t_p + self.find_nB0() * self.find_LB() / self.t_n)

    def find_IR0(self):
        return np.pi * (self.d_C ** 2) / 4 * self.q * \
            (self.find_nB0() * self.find_LB() / self.t_n + self.find_pC0() * self.find_LC() / self.t_p)

    def find_IF(self):
        return self.find_IF0() * (np.exp(self.V_BE / self.k / self.T) - 1)

    def find_IR(self):
        return self.find_IR0() * (np.exp(-self.V_CB / self.k / self.T) - 1)
    
    def find_alfaR(self):
        return (self.I_E + self.find_IF()) / self.find_IR()

    def find_alfaF(self):
        return self.find_alfaR() * self.find_IR0() / self.find_IF0()

class Draw_graph(Culculation_param):
    def draw_N_x(self):
        x_list = list()
        N_list = list()

        for x in np.arange(0, self.x_E + self.x_B + self.x_C, 1E-6):
            if x <= self.x_E - self.find_xdep_E():
                x_list.append(x)
                N_list.append(self.find_pE_x(x))
            
            elif x >= self.find_xdepB1() + self.x_E and x <= self.x_E + self.x_B - self.find_xdepB2():
                x_list.append(x)
                N_list.append(self.find_n_B_X(x))

            elif x >= self.x_E + self.x_B + self.find_xdep_C():
                x_list.append(x)
                N_list.append(self.find_p_C_x(x))

            else:
                x_list.append(x)
                N_list.append(0)
            
        fig, axes = plt.subplots()

        axes.plot(x_list, N_list, color='b', label='N(x)', linewidth=1.5)
        plt.ylabel('N, см^(-3)')
        plt.xlabel('x, см')
        plt.legend(loc=5)
        axes.set(ylim=(0))

        plt.show()

def main():
    Ge = Culculation_param()
    draw = Draw_graph()
    '''
    print('Задание 2')
    print(f'fi_EB = {Ge.find_fi_EB()}')
    print(f'fi_CB = {Ge.find_fi_CB()}')
    print(f'WdepEB = {Ge.find_Wdep_EB()}')
    print(f'xdep_E = {Ge.find_xdep_E()}')
    print(f'xdepB1 = {Ge.find_xdepB1()}')
    print(f'WdepBC = {Ge.find_Wdep_BC()}')
    print(f'xdep_C = {Ge.find_xdep_C()}')
    print(f'xdepB2 = {Ge.find_xdepB2()}')
    print(f'WdepEB_r = {Ge.find_Wdep_EB_r()}')
    print(f'xdep_E_r = {Ge.find_xdep_E_r()}')
    print(f'xdepB1_r = {Ge.find_xdepB1_r()}')
    print(f'WdepBC_r = {Ge.find_Wdep_BC_r()}')
    print(f'xdep_C_r = {Ge.find_xdep_C_r()}')
    print(f'xdepB2_r = {Ge.find_xdepB2_r()}')
    print(f'dF_E={Ge.find_dF_E()}')
    print(f'dF_B={Ge.find_dF_B()}')
    print(f'dF_C={Ge.find_dF_C()}')
    
    print('Задание 3')
    print(f'pE0 = {Ge.find_pE0()}')
    print(f'nB0 = {Ge.find_nB0()}')
    print(f'pC0 = {Ge.find_pC0()}')
    print(f'pE = {Ge.find_pE()}')
    print(f'nB1 = {Ge.find_nB1()}')
    print(f'LE = {Ge.find_LE()}')
    print(f'LB = {Ge.find_LB()}')
    print(f'LC = {Ge.find_LC()}')
    #draw.draw_N_x()
    

    print('Задание 4')
    print(f'gamma = {Ge.find_gamma()}')
    print(f'alfaT = {Ge.find_alfaT()}')
    print(f'Ir0 = {Ge.find_Ir0()}')
    print(f'Is0 = {Ge.find_Is0()}')
    print(f'delta = {Ge.find_delta()}')
    print(f'alfa = {Ge.find_alfa()}')
    print(f'beta = {Ge.find_beta()}')
    
    
    print('Задание 5')
    print(f'Vp = {Ge.find_Vp()}')
    

    print('Задание 6')
    print(f'VA = {Ge.find_VA()}')
    '''

    print('Задание 7')
    print(f'IF0 = {Ge.find_IF0()}')
    print(f'IR0 = {Ge.find_IR0()}')
    print(f'IF = {Ge.find_IF()}')
    print(f'IR = {Ge.find_IR()}')
    print(f'alfaR = {Ge.find_alfaR()}')
    print(f'alfaF = {Ge.find_alfaF()}')

if __name__ == "__main__":
    main()
