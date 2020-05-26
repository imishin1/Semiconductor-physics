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
        self.mu_n_300_prim = 4.5E3 # см2/В/с
        self.mu_p_300_prim = 4E2 # см2/В/с
        self.Eg_kof = 0.785 # эВ
        self.Eg_alfa = -4.2E-4 # эВ/K
        self.N_d = 5.5E14 # см-3
        self.N_a = 8.6E17 # см-3
        self.tn = 4.2E-5 # с
        self.tp = 1.1E-5 # с
        self.d = 6.8E-2 # см
        self.Vr = 4 # В
        self.R = 3500 # Ом
        self.Vps = 3.3 # В
        self.E_a = 0.01 # эВ
        self.E_d = 0.01 # эВ
        self.g = 2
        self.epsilond = 16.3

class Culculation_param():
    def __init__(self):
        Physical_quantities.__init__(self)
        Initial_param.__init__(self)

    def find_N_a_critical(self):
        return (10 ** 22.5) * (self.m_dp ** 1.5) * (self.E_a ** 1.5)
    
    def find_N_d_critical(self):
        return (10 ** 22.5) * (self.m_dn ** 1.5) * (self.E_d ** 1.5)

    def find_N_c(self, temperature):
        return 4.831e15 * (self.m_dn ** 1.5) * (temperature ** 1.5)
    
    def find_N_v(self, temperature):
        return 4.831e15 * (self.m_dp ** 1.5) * (temperature ** 1.5)

    def find_E_g(self, temperature):
        return self.Eg_kof - self.Eg_alfa * temperature

    def find_ni(self, temperature):
        return (self.find_N_c(temperature) * self.find_N_v(temperature)) ** 0.5 * \
            np.exp(-self.find_E_g(temperature) / 2 / self.k / temperature)
    
    def find_Nb(self):
        return self.N_a * self.N_d / (self.N_a + self.N_d)
    
    def find_Ec_F(self, temperature):
        return self.k * temperature * np.log(self.find_N_c(temperature) / self.N_d)

    def find_F_Ev(self, temperature):
        return self.k * temperature * np.log(self.find_N_v(temperature) / self.N_a)

    def find_fi_k(self, temperature):
        return np.log(self.N_d * self.N_a / (self.find_ni(temperature) ** 2)) * self.k * temperature

    def find_W_dep(self, temperature):
        return (2 * self.epsilond * self.epsilond_0 / self.q * self.find_fi_k(temperature) / \
            self.find_Nb()) ** 0.5

    def find_x_n(self, temperature):
        return self.N_a / (self.N_a + self.N_d) * self.find_W_dep(temperature)

    def find_x_p(self, temperature):
        return self.find_W_dep(temperature) - self.find_x_n(temperature)

    def find_E_p(self, temperature, x):
        return - self.q * self.N_a / (self.epsilond * self.epsilond_0) * (x + self.find_x_p(temperature))

    def find_E_n(self, temperature, x):
        return - self.q * self.N_d / (self.epsilond * self.epsilond_0) * (self.find_x_n(temperature) - x)

    def find_U_p(self, temperature, x):
        return self.q * self.N_a / (2 * self.epsilond * self.epsilond_0) * \
            ((x + self.find_x_p(temperature)) ** 2)

    def find_U_n(self, temperature, x):
        return self.q * self.N_d / (self.epsilond * self.epsilond_0) * \
            (self.find_x_n(temperature) * x - (x ** 2) /2) + \
            self.q * self.N_a / (2 * self.epsilond * self.epsilond_0) * (self.find_x_p(temperature) ** 2)

    def find_W_dep_obr(self, temperature):
        return (2 * self.epsilond * self.epsilond_0 / self.q * (self.find_fi_k(temperature)+self.Vr) / \
            self.find_Nb()) ** 0.5 

    def find_x_n_obr(self, temperature):
        return self.N_a / (self.N_a + self.N_d) * self.find_W_dep_obr(temperature)

    def find_x_p_obr(self, temperature):
        return self.find_W_dep_obr(temperature) - self.find_x_n_obr(temperature)

    def find_E_p_obr(self, temperature, x):
        return - self.q * self.N_a / (self.epsilond * self.epsilond_0) * (x + self.find_x_p_obr(temperature))

    def find_E_n_obr(self, temperature, x):
        return - self.q * self.N_d / (self.epsilond * self.epsilond_0) * (self.find_x_n_obr(temperature) - x)

    def find_U_p_obr(self, temperature, x):
        return self.q * self.N_a / (2 * self.epsilond * self.epsilond_0) * \
            ((x + self.find_x_p_obr(temperature)) ** 2)

    def find_U_n_obr(self, temperature, x):
        return self.q * self.N_d / (self.epsilond * self.epsilond_0) * \
            (self.find_x_n_obr(temperature) * x - (x ** 2) /2) + \
            self.q * self.N_a / (2 * self.epsilond * self.epsilond_0) * (self.find_x_p_obr(temperature) ** 2)

    def find_S(self):
        return np.pi * self.d / 4
    
    def find_Cb(self, temperature, Vr):
        return self.find_S() * ((self.epsilond * self.epsilond_0 * \
            self.q * self.N_a * self.N_d / 2 / (self.N_d + self.N_a) / \
            (self.find_fi_k(temperature)-Vr)) ** 0.5)

    def find_jns(self, temperature):
        return self.q * (self.find_ni(temperature) ** 2) / self.N_a / self.tn * \
            ((self.k * temperature * self.mu_n_300_prim * self.tn)  ** 0.5)

    def find_jps(self, temperature):
        return self.q * (self.find_ni(temperature) ** 2) / self.N_d / self.tp * \
            ((self.k * temperature * self.mu_p_300_prim * self.tp)  ** 0.5)

    def find_Ins(self, temperature):
        return self.find_jns(temperature) * self.find_S()

    def find_Ips(self, temperature):
        return self.find_jps(temperature) * self.find_S()

    def find_Is(self, temperature):
        return self.find_Ins(temperature) + self.find_Ips(temperature)

    def find_Irec(self, temperature, U):
        up_kof = self.q * self.find_ni(temperature) * ((2 * self.epsilond * self.epsilond_0 / self.q *\
            (self.find_fi_k(temperature) + U) * (self.N_d + self.N_a) / self.N_a / self.N_d) ** 0.5)
        down_kof = self.tn + self.tp
        return up_kof / down_kof * self.find_S() * np.exp(U / 2 / self.k / temperature)

    def find_I0(self, temperature, U):
        return self.find_Is(temperature) * (np.exp(U / self.k / temperature) - 1)

    def find_Ipr(self, temperature, U):
        return self.find_Irec(temperature, U) + self.find_I0(temperature, U)

    def find_Igen(self, temperature, U):
        up_kof = self.q * self.find_ni(temperature) * ((2 * self.epsilond * self.epsilond_0 / self.q *\
            (self.find_fi_k(temperature) - U) * (self.N_d + self.N_a) / self.N_a / self.N_d) ** 0.5)
        down_kof = self.tn + self.tp
        return up_kof / down_kof * self.find_S()
  
    def find_Iobr(self, temperature, U):
        return self.find_I0(temperature, U) - self.find_Igen(temperature, U)

    def find_Eq(self, temperature, U):
        return U + self.R * self.find_Is(np.exp(U / self.k / temperature) - 1) - self.Vps

    def find_Ud(self, temperature):
        def find_Eq(U):
            return U + self.R * self.find_Is(np.exp(U / self.k / temperature) - 1) - self.Vps
        Ud = fsolve(find_Eq, 2)
        return Ud[0]
    
    def find_Ur(self, temperature):
        return self.Vps - self.find_Ud(temperature)

    def find_Id(self, temperature):
        return self.find_Ur(temperature) / self.R

    def find_Rd(self, temperature):
        return self.find_Ud(temperature) / self.find_Id(temperature)

    def find_Cbar(self, temperature):
        return self.find_S() * ((self.epsilond * self.epsilond_0 / 2 * self.q *\
            self.N_a * self.N_d / (self.N_d + self.N_a) / \
            (self.find_fi_k(temperature) - self.find_Ud(temperature))) ** 0.5)

    def find_Cdif(self, temperature):
        return 1 / 2 / temperature / self.k * np.exp(self.find_Ud(temperature) / temperature / self.k)\
            * (self.find_Ins(temperature) * self.tn + self.find_Ips(temperature) * self.tp)

class Draw_graph(Culculation_param):
    def draw_electric_field(self, temperature):
        x_list = list()
        E_list = list()

        for x in np.arange (-1 * self.find_x_p_obr(temperature), 0, 1E-9):
            x_list.append(x)
            E_list.append(self.find_E_p(temperature, x))
        for x in np.arange (0, self.find_x_n_obr(temperature) , 1E-9):
            x_list.append(x)
            E_list.append(self.find_E_n(temperature, x))

        fig, axes = plt.subplots()

        axes.plot(x_list, E_list, color='b', label='E(x)', linewidth=1.5)
        #axes.set(xlim=(-1E-7, 1E-5))

        plt.title('Распеделение электрического поля в p-n переходе')
        plt.xlabel('х')
        plt.ylabel('Е')
        plt.legend(loc=5)
        
        plt.show()
        
    def draw_potential(self, temperature):
        x_list = list()
        U_list = list()

        for x in np.arange (-1 * self.find_x_p_obr(temperature), 0, 1E-7):
            x_list.append(x)
            U_list.append(self.find_U_p(temperature, x))
        for x in np.arange (0, self.find_x_n_obr(temperature) , 1E-7):
            x_list.append(x)
            U_list.append(self.find_U_n(temperature, x))

        fig, axes = plt.subplots()

        axes.plot(x_list, U_list, color='b', label='U(x)', linewidth=1.5)
        axes.set(xlim=(0))
        axes.set(ylim=(0))
        plt.title('Распеделение электрического поля в p-n переходе')
        plt.xlabel('х')
        plt.ylabel('U')
        plt.legend(loc=5)
        
        plt.show()

    def draw_electric_field_obr(self, temperature):
        x_list = list()
        E_list = list()

        for x in np.arange (-1 * self.find_x_p_obr(temperature), 0, 1E-9):
            x_list.append(x)
            E_list.append(self.find_E_p_obr(temperature, x))
        for x in np.arange (0, self.find_x_n_obr(temperature) , 1E-9):
            x_list.append(x)
            E_list.append(self.find_E_n_obr(temperature, x))

        fig, axes = plt.subplots()

        axes.plot(x_list, E_list, color='b', label='E(x)', linewidth=1.5)
        #axes.set(xlim=(-1E-6, 1E-5))

        plt.title('Распеделение электрического поля в p-n переходе при обратном смещении')
        plt.xlabel('х')
        plt.ylabel('Е')
        plt.legend(loc=5)
        
        plt.show()
        
    def draw_potential_obr(self, temperature):
        x_list = list()
        U_list = list()

        for x in np.arange (-1 * self.find_x_p_obr(temperature), 0, 1E-7):
            x_list.append(x)
            U_list.append(self.find_U_p_obr(temperature, x))
        for x in np.arange (0, self.find_x_n_obr(temperature) , 1E-7):
            x_list.append(x)
            U_list.append(self.find_U_n_obr(temperature, x))

        fig, axes = plt.subplots()

        axes.plot(x_list, U_list, color='b', label='U(x)', linewidth=1.5)
        axes.set(xlim=(0))
        axes.set(ylim=(0))
        plt.title('Распеделение электрического поля в p-n переходе при обратном смещении')
        plt.xlabel('х')
        plt.ylabel('U')
        plt.legend(loc=5)
        
        plt.show()

    def draw_Cb(self, temperature):
        Cb_list = list()
        Vr_list = list()
        for Vr in np.arange(-10, 0, 0.01):
            Vr_list.append(Vr)
            Cb_list.append(self.find_Cb(300, Vr))

        fig, axes = plt.subplots()

        axes.plot(Vr_list, Cb_list, color='b', label='Cb(Vr)', linewidth=1.5)
        plt.xlabel('Vr')
        plt.ylabel('Cb')
        plt.legend(loc=6)
        
        plt.show()

    def draw_VAX_pr(self, temperature):
        U_list = list()
        Ipr_list = list()
        I0_list = list()
        Irec_list = list()
        for U in np.arange(0, 0.4, 0.0005):
            U_list.append(U)
            Ipr_list.append(np.log(self.find_Ipr(temperature, U)))
            I0_list.append(np.log(self.find_I0(temperature, U)))
            Irec_list.append(np.log(self.find_Irec(temperature, U)))

        fig, axes = plt.subplots()

        axes.plot(U_list, Ipr_list, color='b', label='Ipr(U)', linewidth=1.5)
        axes.plot(U_list, I0_list, color='r', label='I0(U)', linewidth=1.5)
        axes.plot(U_list, Irec_list, color='g', label='Irec(U)', linewidth=1.5)
        axes.set(xlim=(0))
        plt.xlabel('U')
        plt.ylabel('I')
        plt.legend(loc=5)
        
        plt.show()

    def draw_VAX_obr(self, temperature):
        U_list = list()
        Iobr_list = list()
        I0_list = list()
        Igen_list = list()
        for U in np.arange(-0.4, 0.1, 0.0005):
            U_list.append(U)
            Iobr_list.append((self.find_Iobr(temperature, U)))
            I0_list.append((self.find_I0(temperature, U)))
            Igen_list.append(-1 * (self.find_Igen(temperature, U)))

        fig, axes = plt.subplots()

        axes.plot(U_list, Iobr_list, color='b', label='Iobr(U)', linewidth=1)
        axes.plot(U_list, I0_list, color='r', label='I0(U)', linewidth=1)
        axes.plot(U_list, Igen_list, color='g', label='-Igen(U)', linewidth=1)
        plt.xlabel('U')
        plt.ylabel('I')
        plt.legend(loc=5)
        
        plt.show()

def main():
    Ge = Culculation_param()
    print('task 1 and 2')
    print(f'N_a_cr = {Ge.find_N_a_critical()}')
    print(f'N_d_cr = {Ge.find_N_d_critical()}')
    print('Невырожден, ой как хорошо')
    print(f'Nc={Ge.find_N_c(300)}')
    print(f'Nv={Ge.find_N_v(300)}')
    print(f'ni={Ge.find_ni(300)}')
    print(f'Nb={Ge.find_Nb()}')
    print(f'Ec-F={Ge.find_Ec_F(300)}')
    print(f'F-Ev={Ge.find_F_Ev(300)}')
    print(f'fik={Ge.find_fi_k(300)}')
    print(f'Wdep={Ge.find_W_dep(300)}')
    print(f'xn={Ge.find_x_n(300)}')
    print(f'xp={Ge.find_x_p(300)}')
    print('task 4 and 5')
    print(f'Wdep_obr={Ge.find_W_dep_obr(300)}')
    print(f'xn_obr={Ge.find_x_n_obr(300)}')
    print(f'xp_obr={Ge.find_x_p_obr(300)}')
    print('task 7')
    print(f'S={Ge.find_S()}')
    print('task 8')
    print(f'jns={Ge.find_jns(300)}')
    print(f'jps={Ge.find_jps(300)}')
    print(f'Ins={Ge.find_Ins(300)}')
    print(f'Ips={Ge.find_Ips(300)}')
    print(f'Is={Ge.find_Is(300)}')
    print(f'Ud={Ge.find_Ud(300)}')
    print(f'Cbar={Ge.find_Cbar(300)}')
    print(f'Cdif={Ge.find_Cdif(300)}')

    #drawing = Draw_graph()

    #drawing.draw_electric_field(300)
    #drawing.draw_potential(300)

    #drawing.draw_electric_field_obr(300)
    #drawing.draw_potential_obr(300)

    #drawing.draw_Cb(300)

    #drawing.draw_VAX_pr(300)
    #drawing.draw_VAX_obr(300)
    
if __name__ == "__main__":
    main()