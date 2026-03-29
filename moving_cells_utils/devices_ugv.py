import numpy as np
from moving_cells_utils import uopvp_solver as uopvp
import math


class DeviceUGV:
    def __init__(self, env, d_lim: int):

        """
        Initialize with energy costs/limitations and solar cell specifications
        """
        self.env = env
        self.d_max = d_lim
        self.v_max, self.w, self.u_rr = 4, 21, 0.1  # m/min, N, _

        # Solar cell
        self.solar_current, self.solar_voltage, self.solar_area = 6, 18, 1.020 * 0.520  # A, V, m x m
        self.motor_consumption, self.overhead = 120, 30  # W, W
        self.spectral_response = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.23], [0.25], [0.27], [0.29], [0.31], [0.33], [0.3376], [0.3452], [0.3529],
                                           [0.3605], [0.3681], [0.3757], [0.3833], [0.3910], [0.3986], [0.4062],
                                           [0.4138], [0.4214], [0.4290], [0.4367], [0.4443], [0.4595], [0.4694],
                                           [0.4824], [0.4976], [0.5174], [0.5263], [0.5433], [0.5586], [0.5647],
                                           [0.5695], [0.5814], [0.5910], [0.5948], [0.5986], [0.6024], [0.6119],
                                           [0.6271], [0.6393], [0.6427], [0.6274], [0.6107], [0.5714], [0.5321],
                                           [0.4830], [0.4634], [0.4437], [0.4339], [0.4202], [0.3986], [0.3652],
                                           [0.3357], [0.3092], [0.2179], [0.1589], [10.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                                           [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

        self.router = uopvp.UOPVPSolver(env, self)
        self.position = 0
        self.curr_location = self.router.current_route[self.position]
        self.next_location = self.router.current_route[self.position]
        self.service_intervals = [0] * len(self.router.current_route)
        time = 0
        for interval in range(len(self.router.current_service_interval)):
            if interval > 0:
                time += self.router.current_route[interval] + self.router.current_t_route[interval-1]
            else:
                time += self.router.current_route[interval]
            self.service_intervals[interval] = time
        self.current_service_interval = self.service_intervals[self.position]


    def find_power(self, x: int, y: int, time: int):
        spectra = self.env.get_spectrum(x, y, time)
        interference = self.env.get_obfuscation(x, y, time)
        integral_content = np.nan_to_num(spectra['poa_global'], nan=0) * self.spectral_response
        cell_current = np.trapz(integral_content, spectra['wavelength'], axis=0)
        a = time / 60 + 2
        alpha = abs(104 - 65 * a + 47 * pow(a, 2) - 12 * pow(a, 3) + pow(a, 4))
        power = abs(alpha / 100) * interference * cell_current * self.solar_area  # W
        return power


    def get_constraints(self):
        e_max = self.solar_current * self.solar_voltage * 60  # W * s = J
        e_t = e_max * 0.6                                   # J = J
        e_move = self.motor_consumption * 60                  # W * s = J
        return e_t, e_max, e_move, self.v_max, self.d_max


    def _move_to_point(self):
        travel_dist = math.sqrt((self.next_location[0] - self.curr_location[0]) ** 2 +
                                (self.next_location[1] - self.curr_location[1]) ** 2)
        if travel_dist < self.v_max:
            t_travelled = travel_dist / self.v_max
            e_move = self.motor_consumption * t_travelled
            self.curr_location = self.next_location
        else:
            e_move = self.motor_consumption * 60
            self.est_point = [self.curr_location[0] + (self.v_max/travel_dist) *
                              (self.next_location[0] - self.curr_location[0]),
                              self.curr_location[1] + (self.v_max/travel_dist) *
                              (self.next_location[1] - self.curr_location[1])]
            self.curr_location = [round(self.est_point[0]), round(self.est_point[1])]
        return e_move


    def step(self, t: int):
        """
        Harvest energy using solar cells and move UGV
        """
        if t >= self.current_service_interval:
            self.position += 1
            self.next_location = self.router.current_route[self.position]
            self.current_service_interval = self.service_intervals[self.position]

        e_move = 0  # W
        if self.next_location != self.curr_location:
            e_move = self._move_to_point()
        e_harvest = self.harvest(t)
        e_overhead = self.overhead * 60

        return e_harvest, e_move, e_overhead


    def try_harvest(self, x: int, y: int, t: int):
        """
        Try energy harvesting at a coordinate (x, y)  and time then return energy
        """
        return self.find_power(x, y, t) * 60


    def harvest(self, t: int):
        """
        Harvest energy from solar cells
        """
        return self.find_power(self.curr_location[0], self.curr_location[1], t) * 60
