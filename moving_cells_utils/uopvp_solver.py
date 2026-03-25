import math
import copy
import random
import os
import datetime
import csv
import atexit
import time


def sum_sqrt_diff_sq(x: int, stop: int):
    summation = 0
    for i in range(x):
        if x == stop:
            break
        summation += math.sqrt(x**2 - (i+1)**2)
    return summation


class UOPVPSolver:
    def __init__(self, env, agent):
        """
        Set UOPVP variables, set initial route, find initial window, call further optimization
        """
        self.env = env  # Environment class object
        self.agent = agent  # UGV class object
        self.num_tries = 10
        self.planner_limit = 10
        self.optimizer_minimum, self.optimizer_steps, self.optimizer_weight = 0.01, 100, 0.9
        self.optimization_steps, self.optimization_threshold = 1000, 0.001

        # Environment Requirements
        self.t_max = env.get_constraints()

        # Agent Requirements
        self.e_t, self.e_max, self.e_move, self.v_max, self.d_max = agent.get_constraints()
        self.move_cost_ratio = self.e_move / self.e_max

        # Find total nuber of vertices
        self.num_vertices = int(1 + 4 * self.d_max + 4 * sum_sqrt_diff_sq(self.d_max, self.d_max))  # Number of Vertices

        """
        Log Files
        """
        # Set directory path
        log_dir = "moving_cells_utils/logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_str = ".csv"
        date_time = datetime.datetime.now()
        self.change_num = 0
        # Route logging
        route_log = ("solver_routes_" + date_time.strftime("%d") + "_" +
                        date_time.strftime("%m") + csv_str)
        route_logfile = os.path.join(log_dir, route_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.route_file = open(route_logfile, mode='a', newline='', encoding='utf-8')
        self.route_writer = csv.writer(self.route_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(route_logfile) == 0:
            self.route_writer.writerow(["change", "route"])
            self.route_file.flush()

        # Transition logging
        t_route_log = ("solver_transitions_" + date_time.strftime("%d") + "_" +
                     date_time.strftime("%m") + csv_str)
        t_route_logfile = os.path.join(log_dir, t_route_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.t_route_file = open(t_route_logfile, mode='a', newline='', encoding='utf-8')
        self.t_route_writer = csv.writer(self.t_route_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(t_route_logfile) == 0:
            self.t_route_writer.writerow(["change", "transitions"])
            self.t_route_file.flush()

        # Service logging
        service_log = ("solver_service_" + date_time.strftime("%d") + "_" +
                       date_time.strftime("%m") + csv_str)
        service_logfile = os.path.join(log_dir, service_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.service_file = open(service_logfile, mode='a', newline='', encoding='utf-8')
        self.service_writer = csv.writer(self.service_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(service_logfile) == 0:
            self.service_writer.writerow(["change", "intervals"])
            self.service_file.flush()

        # Route Scores logging
        scores_log = ("solver_scores_" + date_time.strftime("%d") + "_" +
                      date_time.strftime("%m") + csv_str)
        scores_logfile = os.path.join(log_dir, scores_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.scores_file = open(scores_logfile, mode='a', newline='', encoding='utf-8')
        self.scores_writer = csv.writer(self.scores_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(scores_logfile) == 0:
            self.scores_writer.writerow(["change", "route score", "actual score"])
            self.scores_file.flush()

        # Time logging
        time_log = ("solver_times_" + date_time.strftime("%d") + "_" +
                       date_time.strftime("%m") + csv_str)
        time_logfile = os.path.join(log_dir, time_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.time_file = open(time_logfile, mode='a', newline='', encoding='utf-8')
        self.time_writer = csv.writer(self.time_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(time_logfile) == 0:
            self.time_writer.writerow(["total", "swap", "add", "remove", "replace"])
            self.time_file.flush()

        # Map Logging
        map_log = ("solver_map_" + date_time.strftime("%d") + "_" +
                    date_time.strftime("%m") + csv_str)
        map_logfile = os.path.join(log_dir, map_log)
        # open once, append mode; newline='' avoids blank lines on Windows
        self.map_file = open(map_logfile, mode='a', newline='', encoding='utf-8')
        self.map_writer = csv.writer(self.map_file, delimiter='|')
        # write header only if file is empty
        if os.path.getsize(map_logfile) == 0:
            self.map_writer.writerow(["route"])
            self.map_file.flush()

        start_time = time.perf_counter()
        self.profit_map, self.best_vertices = self.max_profits()
        self.map_writer.writerows(self.profit_map)
        self.map_file.flush()

        self.current_route, self.current_t_route, self.current_route_score = self.initial_planner(self.best_vertices[0])
        self.current_service_interval = self.service_window_optimizer(self.current_route, self.current_t_route)
        self.current_actual_profit = self.find_actual_profits(self.current_route, self.current_t_route,
                                                              self.current_service_interval)

        self.route_writer.writerow([self.change_num, self.current_route])
        self.route_file.flush()
        self.t_route_writer.writerow([self.change_num, self.current_t_route])
        self.t_route_file.flush()
        self.service_writer.writerow([self.change_num, self.current_service_interval])
        self.service_file.flush()
        self.scores_writer.writerow([self.change_num, self.current_route_score, self.current_actual_profit])
        self.scores_file.flush()
        self.change_num += 1
        print(self.change_num)

        (self.current_route, self.current_t_route, self.current_service_interval, self.current_actual_profit,
         self.current_route_score) = (self.optimize_route(self.current_route, self.current_t_route,
                                                          self.current_service_interval, self.current_route_score))

        for root in range(len(self.best_vertices) - 1):
            proposed_route, proposed_t_route, proposed_route_score = self.initial_planner(self.best_vertices[root+1])
            proposed_service_interval = self.service_window_optimizer(proposed_route, proposed_t_route)
            proposed_route, proposed_t_route, proposed_service_interval, proposed_actual_profit, proposed_route_score =\
                (self.optimize_route(proposed_route, proposed_t_route, proposed_service_interval, proposed_route_score))
            if proposed_actual_profit > self.current_actual_profit:
                self.current_route = proposed_route
                self.current_t_route = proposed_t_route
                self.current_service_interval = proposed_service_interval
                self.current_actual_profit = proposed_actual_profit
                self.current_route_score = proposed_route_score

                self.route_writer.writerow([self.change_num, self.current_route])
                self.route_file.flush()
                self.t_route_writer.writerow([self.change_num, self.current_t_route])
                self.t_route_file.flush()
                self.service_writer.writerow([self.change_num, self.current_service_interval])
                self.service_file.flush()
                self.scores_writer.writerow([self.change_num, self.current_route_score, self.current_actual_profit])
                self.scores_file.flush()
                self.change_num += 1
                print(self.change_num)
        execution_time = time.perf_counter() - start_time
        self.time_writer.writerow([execution_time, 0, 0, 0, 0])
        self.time_file.flush()



    def __del__(self):
        try:
            self.route_file.close()
        except Exception:
            pass
        try:
            self.t_route_file.close()
        except Exception:
            pass
        try:
            self.service_file.close()
        except Exception:
            pass
        try:
            self.scores_file.close()
        except Exception:
            pass
        try:
            self.time_file.close()
        except Exception:
            pass
        try:
            self.map_file.close()
        except Exception:
            pass
        atexit.register(lambda: self.route_file and not self.route_file.closed and self.route_file.close())
        atexit.register(lambda: self.t_route_file and not self.t_route_file.closed and self.t_route_file.close())
        atexit.register(lambda: self.service_file and not self.service_file.closed and self.service_file.close())
        atexit.register(lambda: self.scores_file and not self.scores_file.closed and self.scores_file.close())
        atexit.register(lambda: self.time_file and not self.time_file.closed and self.time_file.close())
        atexit.register(lambda: self.map_file and not self.map_file.closed and self.map_file.close())

    def find_profit_and_beta(self, x: int, y: int, t: int):
        energy = self.agent.try_harvest(x, y, t)  # Nonlinear function that determines profit
        powered = 0
        if energy[0] > self.e_t:
            powered = 1
        expected_energy = max((energy[0] - self.e_t) / (self.e_max - self.e_t), 0)
        return expected_energy, powered


    def find_actual_profits(self, route: list, t_route: list, service_intervals: list):
        profit = 0
        current_vertex = 0
        t = 0
        while t < self.t_max:
            energy = self.agent.try_harvest(route[current_vertex][0], route[current_vertex][1], t)
            profit += max((energy[0] - self.e_t) / (self.e_max - self.e_t), 0)
            if t+1 == service_intervals[current_vertex]:
                t += t_route[current_vertex]
                current_vertex += 1
            else:
                t += 1
        return profit


    def find_outage_coefficient(self, pts: list):
        outage_coefficient = 0
        for t in range(self.t_max):
            on_or_off = 0
            for pt in pts:
                energy = self.agent.try_harvest(pt[0], pt[1], t)
                if energy[0] > self.e_t:
                    on_or_off = 1
                    break
            outage_coefficient += on_or_off
        return outage_coefficient / self.t_max


    def max_profits(self):
        best_vertices = [[0, 0, 0]] * self.num_tries
        profit_map = [0] * (1 + 2 * self.env.dim) * (1 + 2 * self.env.dim)
        # Set origin values
        p0, b0 = 0, 0
        for t in range(self.t_max):
            profit, beta = self.find_profit_and_beta(0, 0, t)
            p0 += profit
            b0 += beta
        best_vertices[0] = [0, 0, b0 / self.t_max * p0]
        profit_map[self.get_map_index(0, 0)] = p0
        for i in range(self.d_max):
            for j in range(self.d_max):
                if math.sqrt(i**2 + j**2) < self.d_max:
                    p1, p2, p3, p4 = 0, 0, 0, 0
                    b1, b2, b3, b4 = 0.0, 0.0, 0.0, 0.0
                    for t in range(self.t_max):
                        # Quadrant 1
                        profit, beta = self.find_profit_and_beta(i + 1, j, t)
                        p1 += profit
                        b1 += beta
                        # Quadrant 2
                        profit, beta = self.find_profit_and_beta(-j, i + 1, t)
                        p2 += profit
                        b2 += beta
                        # Quadrant 3
                        profit, beta = self.find_profit_and_beta(-(i + 1), -j, t)
                        p3 += profit
                        b3 += beta
                        # Quadrant 4
                        profit, beta = self.find_profit_and_beta(j, -(i + 1), t)
                        p4 += profit
                        b4 += beta
                    profit_map[self.get_map_index(i + 1, j)] = p1
                    profit_map[self.get_map_index(-j, i + 1)] = p2
                    profit_map[self.get_map_index(-(i + 1), -j)] = p3
                    profit_map[self.get_map_index(j, -(i + 1))] = p4

                    origin_score_pairs = [[i+1, j, p1*b1/self.t_max],
                                          [-j, i+1, p2*b2/self.t_max],
                                          [-(i+1), -j, p3*b3/self.t_max],
                                          [j, -(i+1), p4*b4/self.t_max]]
                    print(origin_score_pairs)
                    for origin_pair in origin_score_pairs:
                        test_origin = copy.deepcopy(origin_pair)
                        for vertex in range(self.num_tries):
                            if test_origin[2] > best_vertices[vertex][2]:
                                temp = copy.deepcopy(best_vertices[vertex])
                                best_vertices[vertex] = copy.deepcopy(test_origin)
                                test_origin = copy.deepcopy(temp)

        return profit_map, best_vertices


    def get_map_index(self, x, y):
        return (y + self.env.dim + 1) * (self.env.dim + 1) + (x + self.env.din + 1)


    def initial_planner(self, root: list):
        """
        Using a root vertex, create an initial best route
        """
        route = [[root[0], root[1]]]
        current_best_route = copy.deepcopy(route)
        t_route = []
        current_best_t_route = copy.deepcopy(t_route)
        route_score = root[2]
        current_best_route_score = root[2]
        searches = 0
        while sum(t_route) <= 0.2 * self.t_max and searches < self.planner_limit:
            print(searches)
            searches += 1
            last_node = copy.deepcopy(current_best_route)[-1]
            for x in range(-2*self.v_max, 2*self.v_max):
                y_max = int(math.sqrt((2*self.v_max)**2 - (x+1)**2) + 1)
                for y in range(-y_max, y_max):
                    if (math.sqrt(x**2 + y**2) >= self.v_max
                            and math.sqrt((x+last_node[0])**2 + (y+last_node[1])**2) <= self.d_max):
                        proposed_node = [[last_node[0]+x, last_node[1]+y]]
                        proposed_route = copy.deepcopy(route)
                        proposed_route.append(proposed_node)
                        proposed_t_route = copy.deepcopy(current_best_t_route)
                        proposed_t_route.append(int(math.sqrt(x ** 2 + y ** 2) / self.v_max))
                        
                        travel_cost = sum(proposed_t_route) * self.move_cost_ratio
                            
                        outage_coefficient = self.find_outage_coefficient(proposed_route)
                        average_profit = 0
                        for node in proposed_route:
                            average_profit += self.profit_map[self.get_map_index(node[0], node[1])]
                        
                        proposed_route_score = outage_coefficient * average_profit / len(proposed_route) - travel_cost
                        if proposed_route_score > current_best_route_score:
                            current_best_route = copy.deepcopy(proposed_route)
                            current_best_t_route = copy.deepcopy(proposed_t_route)
                            current_best_route_score = proposed_route_score
            
            route = copy.deepcopy(current_best_route)
            t_route = copy.deepcopy(current_best_t_route)
            route_score = current_best_route_score
            
        return route, t_route, route_score


    def service_window_optimizer(self, route: list, t_route: list):
        """
        Using a route, find optimal service window
        """
        available_time = self.t_max - sum(t_route)
        service_intervals = []
        for m in range(len(t_route)):
            s = (available_time - sum(service_intervals[:m])) / (len(route) - m)
            left, right, t_lr = route[m], route[m+1], t_route[m]

            time_to_m = sum(service_intervals[:m]) + sum(t_route[:m])
            start, stop = time_to_m, time_to_m+t_lr
            profit_avg_i = sum(self.find_profit_and_beta(route[m][0], route[m][1], t)[0]
                               for t in range(start+s, stop+s))/t_lr
            profit_avg_j = sum(self.find_profit_and_beta(route[m + 1][0], route[m + 1][1], t)[0]
                               for t in range(start+s, stop+s))/t_lr
            profit_chg_i = (self.find_profit_and_beta(route[m][0], route[m][1], stop + s)[0] -
                            self.find_profit_and_beta(route[m][0], route[m][1], start + s)[0])
            profit_chg_j = (self.find_profit_and_beta(route[m + 1][0], route[m + 1][1], stop + s)[0] -
                            self.find_profit_and_beta(route[m + 1][0], route[m + 1][1], start + s)[0])

            step = 0
            while (not ((profit_avg_i - profit_avg_j)**2 < self.optimizer_minimum and profit_chg_i - profit_chg_j < 0)
                   and step < self.optimizer_steps and s > 0):
                print(step)
                s = max(0, s + int(self.optimizer_weight * (profit_avg_i - profit_avg_j) +
                                   self.optimizer_weight**2 * (profit_chg_i**2 - profit_chg_j**2)))
                profit_avg_i = sum(self.find_profit_and_beta(route[m][0], route[m][1], t)[0]
                                   for t in range(start + s, stop + s)) / t_lr
                profit_avg_j = sum(self.find_profit_and_beta(route[m + 1][0], route[m + 1][1], t)[0]
                                   for t in range(start + s, stop + s)) / t_lr
                profit_chg_i = (self.find_profit_and_beta(route[m][0], route[m][1], stop + s)[0] -
                                self.find_profit_and_beta(route[m][0], route[m][1], start + s)[0])
                profit_chg_j = (self.find_profit_and_beta(route[m + 1][0], route[m + 1][1], stop + s)[0] -
                                self.find_profit_and_beta(route[m + 1][0], route[m + 1][1], start + s)[0])
                step += 1
            service_intervals.append(s)
        service_intervals.append(available_time - sum(service_intervals))
        return service_intervals


    @staticmethod
    def swap(route: list, t_route: list, a: int, b: int):
        """
        SWAP two vertices in the route
        """
        if a < 0 or b < 0 or a >= len(route) or b >= len(route) or a == b:
            return route
        alt_route, alt_t_route = copy.deepcopy(route), copy.deepcopy(t_route)

        a_before, a_after = None if a == 0 else alt_route[a-1], None if a == len(alt_route)-1 else alt_route[a+1]
        b_before, b_after = None if b == 0 else alt_route[b-1], None if b == len(alt_route)-1 else alt_route[b+1]

        t_a_before = (0 if b == 0 else
                      math.sqrt((alt_route[a][0]-b_before[0])**2+(alt_route[a][1]-b_before[1])**2))
        t_a_after = (0 if b == len(alt_route)-1 else
                     math.sqrt((alt_route[a][0]-b_after[0])**2+(alt_route[a][1]-b_after[1])**2))
        t_b_before = (0 if a == 0 else
                      math.sqrt((alt_route[b][0]-a_before[0])**2+(alt_route[b][1]-a_before[1])**2))
        t_b_after = (0 if a == len(alt_route)-1 else
                     math.sqrt((alt_route[b][0]-a_after[0])**2+(alt_route[b][1]-a_after[1])**2))

        if a == 0:
            alt_t_route[a-1] = t_b_before
        if a == len(alt_route)-1:
            alt_t_route[a] = t_b_after
        if b == 0:
            alt_t_route[b - 1] = t_a_before
        if b == len(alt_route)-1:
            alt_t_route[b] = t_a_after

        temp = copy.deepcopy(alt_route)[a]
        alt_route[a] = alt_route[b]
        alt_route[b] = temp

        return alt_route, alt_t_route


    def swap_method(self, route: list, t_route: list, service_intervals: list, route_score: float):
        alt_route, alt_t_route, alt_service, alt_route_score = [copy.deepcopy(route), copy.deepcopy(t_route),
                                                                copy.deepcopy(service_intervals), route_score]
        actual_profit = self.find_actual_profits(alt_route, alt_t_route, alt_service)
        for attempt in range(int(len(alt_route)/2)):
            a, b = random.sample(range(len(alt_route)), k=2)
            proposed_route, proposed_t_route = self.swap(alt_route, alt_t_route, a, b)  # Same route score
            error_threshold = (self.e_t - self.e_max) / self.e_max * sum(proposed_t_route)
            proposed_service = self.service_window_optimizer(proposed_route, proposed_t_route)
            proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route, proposed_service)
            e_ppe = alt_route_score - proposed_actual_profit
            if e_ppe < error_threshold and proposed_actual_profit > actual_profit:
                alt_route = copy.deepcopy(proposed_route)
                alt_t_route = copy.deepcopy(proposed_route)
                alt_service = copy.deepcopy(proposed_service)
                actual_profit = proposed_actual_profit

                self.route_writer.writerow([self.change_num, alt_route])
                self.route_file.flush()
                self.t_route_writer.writerow([self.change_num, alt_t_route])
                self.t_route_file.flush()
                self.service_writer.writerow([self.change_num, alt_service])
                self.service_file.flush()
                self.scores_writer.writerow([self.change_num, proposed_actual_profit, alt_route_score])
                self.scores_file.flush()
                self.change_num += 1
                print(self.change_num)

        return alt_route, alt_t_route, alt_service


    @staticmethod
    def add(route: list, t_route: list, a: list, b: int):
        """
        ADD a vertex to the route
        """
        if b < 0 or b > len(route) or a == route[b]:
            return route
        alt_route, alt_t_route = copy.deepcopy(route), copy.deepcopy(t_route)

        b_before = None if b == 0 else alt_route[b-1]
        if b == 0 and a == b_before:
            return route

        t_a_before = 0 if b == 0 else math.sqrt((a[0]-b_before[0])**2+(a[1]-b_before[1])**2)
        t_a_after = 0 if b == len(route) else math.sqrt((a[0]-route[b][0])**2+(a[1]-route[b][1])**2)

        if b == len(route):
            alt_t_route.append(t_a_before)
            alt_route.append(a)
        else:
            alt_t_route.insert(b, t_a_after)
            if b > 0:
                alt_t_route[b-1] = t_a_before
            alt_route.insert(b, a)

        return alt_route, alt_t_route


    def add_method(self, route: list, t_route: list, service_intervals: list, route_score: float, vertex: int):
        alt_route, alt_t_route, alt_service, alt_route_score = [copy.deepcopy(route), copy.deepcopy(t_route),
                                                                copy.deepcopy(service_intervals), route_score]
        actual_profit = self.find_actual_profits(alt_route, alt_t_route, alt_service)
        changed = False
        time_swap = 0
        for attempt in range(int(self.v_max + sum_sqrt_diff_sq(self.v_max, self.v_max))):
            x = random.sample(range(-self.v_max, self.v_max), k=1)
            y_max = int(math.sqrt(self.v_max**2 - (x+1)**2)+1)
            y = random.sample(range(-y_max, y_max), k=1)
            proposed_pt = [alt_route[vertex][0]+x, alt_route[vertex][1]+y]
            if math.sqrt(proposed_pt[0]**2 + proposed_pt[1]**2) <= self.d_max:
                proposed_route, proposed_t_route = self.add(alt_route, alt_t_route, proposed_pt, vertex)
                average_profit = 0
                for node in proposed_route:
                    average_profit += self.profit_map[self.get_map_index(node[0], node[1])]
                proposed_route_score = (self.find_outage_coefficient(proposed_route) *
                                        average_profit / len(proposed_route) -
                                        sum(proposed_t_route) * self.move_cost_ratio)

                if proposed_route_score >= alt_route_score:
                    error_threshold = (self.e_t - self.e_max) / self.e_max * sum(proposed_t_route)
                    proposed_service = self.service_window_optimizer(proposed_route, proposed_t_route)
                    proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route,
                                                                      proposed_service)
                    e_ppe = proposed_route_score - proposed_actual_profit
                    if e_ppe > error_threshold:
                        time_start = time.perf_counter()
                        proposed_route, proposed_t_route, proposed_service = (
                            self.swap_method(proposed_route, proposed_t_route, proposed_service, proposed_route_score))
                        time_swap = time.perf_counter() - time_start
                    proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route,
                                                                      proposed_service)
                    if proposed_actual_profit > actual_profit:
                        alt_route = copy.deepcopy(proposed_route)
                        alt_t_route = copy.deepcopy(proposed_route)
                        alt_service = copy.deepcopy(proposed_service)
                        alt_route_score = proposed_route_score
                        changed = True

                        self.route_writer.writerow([self.change_num, alt_route])
                        self.route_file.flush()
                        self.t_route_writer.writerow([self.change_num, alt_t_route])
                        self.t_route_file.flush()
                        self.service_writer.writerow([self.change_num, alt_service])
                        self.service_file.flush()
                        self.scores_writer.writerow([self.change_num, proposed_actual_profit, alt_route_score])
                        self.scores_file.flush()
                        self.change_num += 1
                        print(self.change_num)
                        break
        return alt_route, alt_t_route, alt_service, alt_route_score, changed, time_swap


    @staticmethod
    def remove(route: list, t_route: list, a: int):
        """
        REMOVE a vertex from the route
        """
        if a < 0 or a >= len(route):
            return route
        alt_route, alt_t_route = copy.deepcopy(route), copy.deepcopy(t_route)

        a_before, a_after = None if a == 0 else alt_route[a - 1], None if a == len(alt_route)-1 else alt_route[a + 1]
        if a == len(alt_route)-1:
            alt_route.pop(), alt_t_route.pop()
        else:
            if a > 0:
                t_before_to_after = math.sqrt((a_before[0]-a_after[0])**2+(a_before[1]-a_after[1])**2)
                alt_t_route[a-1] = t_before_to_after

            alt_t_route.pop(a)
            alt_route.pop(a)

        return alt_route, alt_t_route


    def remove_method(self, route: list, t_route: list, service_intervals: list, route_score: float):
        alt_route, alt_t_route, alt_service, alt_route_score = [copy.deepcopy(route), copy.deepcopy(t_route),
                                                                copy.deepcopy(service_intervals), route_score]
        actual_profit = self.find_actual_profits(alt_route, alt_t_route, alt_service)
        vertices = random.sample(range(len(route)-1), k=int(len(route)/2))
        changed = False
        time_swap = 0
        for vertex in vertices:
            proposed_route, proposed_t_route = self.remove(alt_route, alt_t_route, vertex)
            average_profit = 0
            for node in proposed_route:
                average_profit += self.profit_map[self.get_map_index(node[0], node[1])]
            proposed_route_score = (self.find_outage_coefficient(proposed_route) *
                                    average_profit / len(proposed_route) -
                                    sum(proposed_t_route) * self.move_cost_ratio)
            if proposed_route_score >= alt_route_score:
                error_threshold = (self.e_t - self.e_max) / self.e_max * sum(proposed_t_route)
                proposed_service = self.service_window_optimizer(proposed_route, proposed_t_route)
                proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route,
                                                                  proposed_service)
                e_ppe = proposed_route_score - proposed_actual_profit
                if e_ppe > error_threshold:
                    time_start = time.perf_counter()
                    proposed_route, proposed_t_route, proposed_service = (
                        self.swap_method(proposed_route, proposed_t_route, proposed_service, proposed_route_score))
                    time_swap = time.perf_counter() - time_start
                proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route,
                                                                  proposed_service)
                if proposed_actual_profit > actual_profit:
                    alt_route = copy.deepcopy(proposed_route)
                    alt_t_route = copy.deepcopy(proposed_route)
                    alt_service = copy.deepcopy(proposed_service)
                    alt_route_score = proposed_route_score
                    changed = True

                    self.route_writer.writerow([self.change_num, alt_route])
                    self.route_file.flush()
                    self.t_route_writer.writerow([self.change_num, alt_t_route])
                    self.t_route_file.flush()
                    self.service_writer.writerow([self.change_num, alt_service])
                    self.service_file.flush()
                    self.scores_writer.writerow([self.change_num, proposed_actual_profit, alt_route_score])
                    self.scores_file.flush()
                    self.change_num += 1
                    print(self.change_num)
                    break
        return alt_route, alt_t_route, alt_service, alt_route_score, changed, time_swap


    @staticmethod
    def replace(route: list, t_route: list, a: list, b: int):
        """
        REPLACE a vertex on the route with a nearby vertex
        """
        if b < 0 or b >= len(route) or a == route[b]:
            return route
        alt_route, alt_t_route = copy.deepcopy(route), copy.deepcopy(t_route)

        b_before, b_after = None if b == 0 else alt_route[b - 1], None if b == len(alt_route)-1 else alt_route[b + 1]
        if (b == 0 and b_before == a) or (b == len(alt_route)-1 and b_after == a):
            return route

        t_a_before = 0 if b == 0 else math.sqrt((a[0]-b_before[0])**2+(a[1]-b_before[1])**2)
        t_a_after = 0 if b == len(alt_route)-1 else math.sqrt((a[0]-b_after[0])**2+(a[1]-b_after[1])**2)
        if b == len(alt_route)-1:
            alt_route.pop(), alt_t_route.pop()
            alt_route.append(a), alt_t_route.append(t_a_before)
        else:
            if b > 0:
                alt_t_route[b - 1] = t_a_before

            alt_t_route[b] = t_a_after
            alt_route[b] = a

        return alt_route, alt_t_route


    def replace_method(self, route: list, t_route: list, service_intervals: list, route_score: float, vertex: int):
        alt_route, alt_t_route, alt_service, alt_route_score = [copy.deepcopy(route), copy.deepcopy(t_route),
                                                                copy.deepcopy(service_intervals), route_score]
        actual_profit = self.find_actual_profits(alt_route, alt_t_route, alt_service)
        changed = False
        time_swap = 0
        for attempt in range(int(self.v_max + sum_sqrt_diff_sq(self.v_max, self.v_max))):
            x = random.sample(range(-self.v_max, self.v_max), k=1)
            y_max = int(math.sqrt(self.v_max ** 2 - (x + 1) ** 2) + 1)
            y = random.sample(range(-y_max, y_max), k=1)
            proposed_pt = [alt_route[vertex][0] + x, alt_route[vertex][1] + y]
            if math.sqrt(proposed_pt[0]**2 + proposed_pt[1]**2) <= self.d_max:
                proposed_route, proposed_t_route = self.replace(alt_route, alt_t_route, proposed_pt, vertex)
                average_profit = 0
                for node in proposed_route:
                    average_profit += self.profit_map[self.get_map_index(node[0], node[1])]
                proposed_route_score = (self.find_outage_coefficient(proposed_route) *
                                        average_profit / len(proposed_route) -
                                        sum(proposed_t_route) * self.move_cost_ratio)
                if proposed_route_score >= alt_route_score:
                    error_threshold = (self.e_t - self.e_max) / self.e_max * sum(proposed_t_route)
                    proposed_service = self.service_window_optimizer(proposed_route, proposed_t_route)
                    proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route,
                                                                      proposed_service)
                    e_ppe = proposed_route_score - proposed_actual_profit
                    if e_ppe > error_threshold:
                        time_start = time.perf_counter()
                        proposed_route, proposed_t_route, proposed_service = (
                            self.swap_method(proposed_route, proposed_t_route, proposed_service, proposed_route_score))
                        time_swap = time.perf_counter() - time_start
                    proposed_actual_profit = self.find_actual_profits(proposed_route, proposed_t_route,
                                                                      proposed_service)
                    if proposed_actual_profit > actual_profit:
                        alt_route = copy.deepcopy(proposed_route)
                        alt_t_route = copy.deepcopy(proposed_route)
                        alt_service = copy.deepcopy(proposed_service)
                        alt_route_score = proposed_route_score
                        changed = True

                        self.route_writer.writerow([self.change_num, alt_route])
                        self.route_file.flush()
                        self.t_route_writer.writerow([self.change_num, alt_t_route])
                        self.t_route_file.flush()
                        self.service_writer.writerow([self.change_num, alt_service])
                        self.service_file.flush()
                        self.scores_writer.writerow([self.change_num, proposed_actual_profit, alt_route_score])
                        self.scores_file.flush()
                        self.change_num += 1
                        print(self.change_num)
                        break
        return alt_route, alt_t_route, alt_service, alt_route_score, changed, time_swap


    def optimize_route(self, route: list, t_route: list, service_intervals: list, route_score: float):
        """
        Using an initial route and service window, find the optimal route and service window for OPVP
        """
        alt_route, alt_t_route, alt_service, alt_route_score = [copy.deepcopy(route), copy.deepcopy(t_route),
                                                                copy.deepcopy(service_intervals), route_score]
        actual_profit = self.find_actual_profits(alt_route, alt_t_route, alt_service)
        e_ppe = alt_route_score - actual_profit
        error_threshold = (self.e_t - self.e_max) / self.e_max * sum(alt_t_route)
        time_swap, time_add, time_remove, time_replace = 0, 0, 0, 0
        full_time_start = time.perf_counter()
        if e_ppe > error_threshold:
            # Internal Exchange: Swap a number of vertices
            time_start = time.perf_counter()
            alt_route, alt_t_route, alt_service = self.swap_method(alt_route, alt_t_route, alt_service, alt_route_score)
            time_swap = time.perf_counter() - time_start

        iterations = 0
        while iterations < self.optimization_steps:
            # External Exchange: Add/Remove/Replace vertices
            # Begin with add method at each vertex
            iterations += 1
            changed = False
            if t_route <= 0.2 * self.t_max and not changed:
                vertex = random.sample(range(len(route) + 1), k=1)
                time_start = time.perf_counter()
                alt_route, alt_t_route, alt_service, alt_route_score, changed, delta_swap = (
                    self.add_method(alt_route, alt_t_route, alt_service, alt_route_score, vertex))
                t_add = time.perf_counter() - time_start - delta_swap
                time_swap += delta_swap
            if len(route) > 1 and not changed:
                time_start = time.perf_counter()
                alt_route, alt_t_route, alt_service, alt_route_score, changed, delta_swap = (
                    self.remove_method(alt_route, alt_t_route, alt_service, alt_route_score))
                time_remove = time.perf_counter() - time_start - delta_swap
                time_swap += delta_swap
            if not changed:
                vertex = random.sample(range(len(route)), k=1)
                time_start = time.perf_counter()
                alt_route, alt_t_route, alt_service, alt_route_score, changed, delta_swap = (
                    self.replace_method(alt_route, alt_t_route, alt_service, alt_route_score, vertex))
                time_replace = time.perf_counter() - time_start - delta_swap
                time_swap += delta_swap

        full_time = time.perf_counter() - full_time_start
        self.time_writer.writerow([full_time, time_swap, time_add, time_remove, time_replace])
        self.time_file.flush()

        return alt_route, alt_t_route, alt_service, alt_route_score, actual_profit
