import numpy as np
from collections import namedtuple
from itertools import combinations
import random


RSSDynamics = namedtuple("RSSDynamics", "ResponseTime alphaLon alphaLat")
alphaLon = namedtuple("alphaLon", "accelMax brakeMax brakeMin")
alphaLat = namedtuple("alphaLat", "accelMax brakeMin")
RSS_restrictions = namedtuple('RSS_restrictions', 'lon_sign lon_val lat_sign lat_val')

class RSS_vehicle(object):
    LONGI_ACC_MIN, LONGI_ACC_MAX = -4, 2
    LAT_ACC_MIN, LAT_ACC_MAX = -4, 4
    # LAT_ACC_MIN, LAT_ACC_MAX = -6, 6
    def __init__(self, vehID):
        self.id = vehID
        self.LENGTH = 5.0
        self.WIDTH = 1.8
        self.heading = 0.0
        self.position = None
        self.speed_long, self.speed_lat = None, None
        self.action = None
        self.information = None
        self.RSS_action = {}
        self.RSS_control = False
        self.highly_critical = False

    def update(self, vehInfo, action = None, highly_critical=False):
        self.heading = (vehInfo[67]-90)/180*np.pi
        head = vehInfo[66]
        center = (head[0]-self.LENGTH/2*np.sin(vehInfo[67]/180*np.pi), head[1]-self.LENGTH/2*np.cos(vehInfo[67]/180*np.pi))
        self.position = np.array(center)
        self.speed_long = vehInfo[64]
        self.speed_lat = vehInfo[50]
        self.lane_index = vehInfo[82]
        if action is not None:
            self.action = action
        self.highly_critical = highly_critical
        self.information = vehInfo


class RSS_model(object):
    """
    RSS model. For more details, please refer to:
    "On a Formal Model of Safe and Scalable Self-driving Cars"
    https://arxiv.org/abs/1708.06374
    """

    def __init__(self, RSS_not_activate_prob=0.):
        # Ego vehicle parameters. All values are absolute values.
        VehicleResponseTime = 0.1  # [s]
        Vehicle_alphaLon_accelMax = 2.
        Vehicle_alphaLon_brakeMax = 4. # 5.
        Vehicle_alphaLon_brakeMin = 4.
        Vehicle_alphaLat_accelMax = 4. # 12.
        Vehicle_alphaLat_brakeMin = 4. # 12.
        assert (Vehicle_alphaLon_brakeMax >= Vehicle_alphaLon_brakeMin)
        VehicleLon = alphaLon(Vehicle_alphaLon_accelMax, Vehicle_alphaLon_brakeMax, Vehicle_alphaLon_brakeMin)
        VehicleLat = alphaLat(Vehicle_alphaLat_accelMax, Vehicle_alphaLat_brakeMin)
        self.VehicleRssDynamics = RSSDynamics(VehicleResponseTime, VehicleLon, VehicleLat)

        # mu lateral speed (fluctuation)
        self.lat_fluctuation = 0.

        # Pre check distance threshold
        self.precheck_threshold = 100  # [m]

        # All pairs of vehicles situations and restrictions.
        self.RSS_state_dict = {}  # Contains all pairs of vehicles situations. key: ego-vehicle id, value: {'dangerousTimeLongi':dangerousTime}
        self.RSS_restrictions_dict = {}  # constraints for each vehicle. key: vehicle id, value: constraints.

        self.lon_extra_buffer = 1.0  # [m]
        self.lat_extra_buffer = 1.0  # [m]

        self.current_time = 0

        self.RSS_not_activate_prob = RSS_not_activate_prob  # There is certain prob at the current moment RSS not actually control vehicle.

    def _same_direction_situation(self, egoVehicle, objectVehicle):
        pass

    def _same_direction_longitudinal_check(self, egoVehicle, objectVehicle):
        pass

    def _same_direction_lateral_check(self, egoVehicle, objectVehicle):
        pass

    def _precheck_sphere_safe(self, egoVehicle, objectVehicle):
        """
        Check two vehicles center distance using sphere, if far away then safe.
        """
        dist = np.linalg.norm(egoVehicle.position - objectVehicle.position)
        if dist >= self.precheck_threshold:
            pair_id = (egoVehicle.id, objectVehicle.id)
            # Update the longitudinal and lateral safety situation of the pair.
            self.RSS_state_dict[pair_id] = {'dangerousTimeLon': None, 'dangerousTimeLat': None}
            return True
        else:
            return False

    def _proper_response_generate(self, egoVehicle, objectVehicle, lon_safe, lat_safe, ego_is_front, ego_is_upper):
        pass

    @staticmethod
    def _pair_id_tuple(c1, c2):
        """
        Since RSS always concern with a pair of vehicles. So (vehicle1, vehicle2) and (vehicle2, vehicle1) is the same.
        This function get the unique pair id for any given two vehicles.
        """
        c1_id, c2_id = str(c1.id), str(c2.id)
        pair_id = (c1_id, c2_id) if c1_id > c2_id else (c2_id, c1_id)
        return pair_id

    @staticmethod
    def _cal_safe_dis_lon_same_direction(v_front, v_rear, rho, a_accel_max, a_brake_min, a_brake_max):
        """
        Definition 1. in RSS.
        All acc here are absolute value
        """
        assert (v_front >= 0 and v_rear >= 0)

        d = v_rear * rho + 0.5 * a_accel_max * (rho ** 2) + (v_rear + rho * a_accel_max) ** 2 / (2 * a_brake_min) - \
            v_front ** 2 / (2 * a_brake_max)
        d_min_lon = np.max([0, d])
        return d_min_lon

    @staticmethod
    def _cal_safe_dis_lat(v_upper, v_lower, rho, mu, a_accel_max_lat, a_brake_min_lat):
        pass

    def _execute_RSS_restriction(self, action, restriction_sign, restriction_value):
        for sign, val in zip(restriction_sign, restriction_value):
            action = self._execute_constraint(action=action, sign=sign, val=val)
        return action

    @staticmethod
    def _execute_constraint(action, sign, val):
        if sign == 'leq':
            new_action = np.clip(action, a_min=-np.inf, a_max=val)
        elif sign == 'geq':
            new_action = np.clip(action, a_min=val, a_max=np.inf)
        else:
            raise ValueError('{0} not supported, please select leq or geq'.format(sign))
        return new_action


class highway_RSS_model(RSS_model):
    """
    The specific RSS model fit for the highway-env environment.
    """

    def __init__(self, RSS_not_activate_prob=0.):
        super().__init__(RSS_not_activate_prob=RSS_not_activate_prob)
        self.cav_veh = RSS_vehicle("CAV")

    @staticmethod
    def _rotate_a_point(x1, y1, x2, y2, angle=0):
        x = (x1 - x2) * np.cos(angle) - (y1 - y2) * np.sin(angle) + x2
        y = (x1 - x2) * np.sin(angle) + (y1 - y2) * np.cos(angle) + y2
        return [x, y]

    def _cal_box(self, x, y, length=5, width=2, angle=0):
        upper_left = self._rotate_a_point(x - 0.5 * length, y + 0.5 * width, x, y, angle=angle)
        lower_left = self._rotate_a_point(x - 0.5 * length, y - 0.5 * width, x, y, angle=angle)
        upper_right = self._rotate_a_point(x + 0.5 * length, y + 0.5 * width, x, y, angle=angle)
        lower_right = self._rotate_a_point(x + 0.5 * length, y - 0.5 * width, x, y, angle=angle)
        xs = [upper_left[0], upper_right[0], lower_right[0], lower_left[0]]
        ys = [upper_left[1], upper_right[1], lower_right[1], lower_left[1]]
        return xs, ys

    def _closest_longitudinal_and_lateral_distance(self, ego, other):
        """
        Signed closet longi (x-axis) and lat (y-axis) distance between two vehicles consider vehicle heading.
        Longitudinal: Positive if two objects not overlap longitudinally, negative otherwise.
        Lateral: Positive if two objects not overlap laterally, negative if overlap.
        """
        ego_x, ego_y = ego.position[0], ego.position[1]
        ego_length, ego_width, ego_heading = ego.LENGTH, ego.WIDTH, ego.heading
        ego_rectangle_pts_x, ego_rectangle_pts_y = self._cal_box(ego_x, ego_y, length=ego_length, width=ego_width, angle=ego_heading)

        other_x, other_y = other.position[0].item(), other.position[1].item()
        other_length, other_width, other_heading = other.LENGTH, other.WIDTH, other.heading
        other_rectangle_pts_x, other_rectangle_pts_y = self._cal_box(other_x, other_y, length=other_length, width=other_width, angle=other_heading)

        # Longitudinal
        if ego_x > other_x:  # ego vehicle is in front
            ego_is_front = True
            closest_longi_dist = min(ego_rectangle_pts_x) - max(other_rectangle_pts_x)
        else:
            ego_is_front = False
            closest_longi_dist = min(other_rectangle_pts_x) - max(ego_rectangle_pts_x)

        # Lat
        if ego_y > other_y:  # ego vehicle is on the left-side (upper) of other
            ego_is_upper = True
            closest_lat_dist = min(ego_rectangle_pts_y) - max(other_rectangle_pts_y)
        else:  # ego vehicle is on the right-side (lower) of other
            ego_is_upper = False
            closest_lat_dist = min(other_rectangle_pts_y) - max(ego_rectangle_pts_y)

        return ego_is_front, closest_longi_dist, ego_is_upper, closest_lat_dist

    def _longitudinal_safety_check(self, egoVehicle, objectVehicle, ego_is_front, closest_longi_dist):
        """
        closest_longi_dist < 0 if already longitudinally overlapped (then definitely dangerous).
        """
        pair_id = self._pair_id_tuple(egoVehicle, objectVehicle)
        lon_safe = True

        ego_v_x = egoVehicle.speed_long
        obj_v_x = objectVehicle.speed_long
        if ego_is_front:
            v_front, v_rear = ego_v_x, obj_v_x
        else:
            v_front, v_rear = obj_v_x, ego_v_x

        if egoVehicle.highly_critical:
            lon_extra_buffer_tmp = 0
        else:
            lon_extra_buffer_tmp = self.lon_extra_buffer
        d_min_lon = lon_extra_buffer_tmp + self._cal_safe_dis_lon_same_direction(v_front=v_front, v_rear=v_rear, rho=self.VehicleRssDynamics.ResponseTime,
                                                                                  a_accel_max=self.VehicleRssDynamics.alphaLon.accelMax,
                                                                                  a_brake_min=self.VehicleRssDynamics.alphaLon.brakeMin,
                                                                                  a_brake_max=self.VehicleRssDynamics.alphaLon.brakeMax)

        if closest_longi_dist > d_min_lon:  # The distance is safe.
            self.RSS_state_dict[pair_id]['dangerousTimeLon'] = None
            return lon_safe
        else:  # Longitudinally dangerous.
            lon_safe = False

            # Update the earliest lon dangerous time if needed.
            earliest_ego_object_lon_dangerous_time_previous = self.RSS_state_dict[pair_id]['dangerousTimeLon']
            if earliest_ego_object_lon_dangerous_time_previous is not None:
                # assert (earliest_ego_object_lon_dangerous_time_previous <= self.current_time)
                pass
            else:
                self.RSS_state_dict[pair_id]['dangerousTimeLon'] = self.current_time
            return lon_safe

    def _lateral_safety_check(self, egoVehicle, objectVehicle, ego_is_upper, closest_lat_dist, closest_longi_dist):
        """
        closest_lat_dist < 0 if already longitudinally overlapped (then definitely dangerous).
        """
        acc_lat_check = 4 # maximum lateral acceleration for safety check

        pair_id = self._pair_id_tuple(egoVehicle, objectVehicle)
        lat_safe = True

        obj_v_y = objectVehicle.speed_lat
        ego_v_y = egoVehicle.speed_lat
        if ego_is_upper:
            v_upper, v_lower = ego_v_y, obj_v_y
        else:
            v_upper, v_lower = obj_v_y, ego_v_y
        
        if egoVehicle.highly_critical:
            lat_extra_buffer_tmp = 0
        else:
            lat_extra_buffer_tmp = self.lat_extra_buffer
        d_min_lat = lat_extra_buffer_tmp + self._cal_safe_dis_lat(v_upper=v_upper, v_lower=v_lower, rho=self.VehicleRssDynamics.ResponseTime, mu=self.lat_fluctuation,
                                                                   a_accel_max_lat=acc_lat_check,
                                                                   a_brake_min_lat=acc_lat_check)
        # if longitudinal overlap based on prediction, lateral becomes more conservative
        overlap_lateral_dist_threshold = 2.0
        pred_time = 1.0 # s
        if egoVehicle.position[0] > objectVehicle.position[0]:
            rangerate = egoVehicle.speed_long-objectVehicle.speed_long
        else:
            rangerate = objectVehicle.speed_long-egoVehicle.speed_long
        predict_longitudinal_dist_change = rangerate*pred_time
        if closest_longi_dist < 0 or closest_longi_dist + predict_longitudinal_dist_change < 0:
            d_min_lat = max(d_min_lat, overlap_lateral_dist_threshold)
        if closest_lat_dist > d_min_lat:
            self.RSS_state_dict[pair_id]['dangerousTimeLat'] = None
            return lat_safe
        else:
            lat_safe = False

            # Update the earliest lat dangerous time if needed.
            earliest_ego_object_lat_dangerous_time_previous = self.RSS_state_dict[pair_id]['dangerousTimeLat']
            if earliest_ego_object_lat_dangerous_time_previous is not None:
                # assert (earliest_ego_object_lat_dangerous_time_previous <= self.current_time)
                pass
            else:
                self.RSS_state_dict[pair_id]['dangerousTimeLat'] = self.current_time
            return lat_safe

    def RSS_act(self, env):
        """
        Generate restrictions for each vehicle in the environment.
        """
        self.current_time = env.time  # Update current time
        self.RSS_restrictions_dict = {}  # Reset the dict
        for veh in env.road.vehicles:
            veh.RSS_dangerous = False

        # Loop to examine any pair of vehicles.
        for vehicle_pair in combinations(env.road.vehicles, r=2):
            egoVehicle, objectVehicle = vehicle_pair[0], vehicle_pair[1]  # Actually it doesn't matter which is ego. Just a name.
            ego_restrictions, obj_restrictions = self.RSS_algorithm(egoVehicle=egoVehicle, objectVehicle=objectVehicle)

            # Update the restriction dict
            self._add_vehicle_restrictions_to_dict(veh=egoVehicle, veh_restrictions=ego_restrictions)
            self._add_vehicle_restrictions_to_dict(veh=objectVehicle, veh_restrictions=obj_restrictions)

    def _add_vehicle_restrictions_to_dict(self, veh, veh_restrictions):
        veh_id = str(veh.id)
        if veh_id not in self.RSS_restrictions_dict:
            self.RSS_restrictions_dict[veh_id] = {'lon_sign': veh_restrictions.lon_sign, 'lon_val': veh_restrictions.lon_val,
                                                  'lat_sign': veh_restrictions.lat_sign, 'lat_val': veh_restrictions.lat_val}
        else:
            self.RSS_restrictions_dict[veh_id]['lon_sign'] += veh_restrictions.lon_sign
            self.RSS_restrictions_dict[veh_id]['lon_val'] += veh_restrictions.lon_val
            self.RSS_restrictions_dict[veh_id]['lat_sign'] += veh_restrictions.lat_sign
            self.RSS_restrictions_dict[veh_id]['lat_val'] += veh_restrictions.lat_val

    def RSS_step(self, env):
        """
        Apply the restrictions on vehicle actions.
        """
        for veh in env.road.vehicles:
            veh_id = str(veh.id)
            veh_RSS_restrictions = self.RSS_restrictions_dict[veh_id]
            acc_x_after_RSS_regulation = self._execute_RSS_restriction(action=veh.action['acc_x'], restriction_sign=veh_RSS_restrictions['lon_sign'],
                                                                restriction_value=veh_RSS_restrictions['lon_val'])
            acc_y_after_RSS_regulation = self._execute_RSS_restriction(action=veh.action['acc_y'], restriction_sign=veh_RSS_restrictions['lat_sign'],
                                                                restriction_value=veh_RSS_restrictions['lat_val'])

            veh.RSS_action['acc_x'] = acc_x_after_RSS_regulation
            veh.RSS_action['acc_y'] = acc_y_after_RSS_regulation

            # Update whether the action has been modified by RSS restrictions.
            rand_number = env.road.np_random.uniform(0, 1)
            veh.RSS_randomly_choose_not_control = False
            if ((acc_x_after_RSS_regulation != veh.action['acc_x']) or (acc_y_after_RSS_regulation != veh.action['acc_y'])):
                if rand_number > self.RSS_not_activate_prob:
                    veh.RSS_control = True
                    veh.action['acc_x'] = acc_x_after_RSS_regulation
                    veh.action['acc_y'] = acc_y_after_RSS_regulation
                else:  # There are some probability that RSS not activate at the moment.
                    veh.RSS_control = False
                    veh.RSS_randomly_choose_not_control = True
            else:
                veh.RSS_control = False


            # Cannot drive backwards
            veh_v_x = veh.speed * np.cos(veh.heading)
            if (veh_v_x <= 0) or (veh_v_x + veh.action['acc_x'] * 1 / env.config['simulation_frequency']) <= 0:
                veh.action['acc_x'] = np.clip(veh.action['acc_x'], 0., np.inf)

            # Physical constraints
            veh.action['acc_x'] = np.clip(veh.action['acc_x'], veh.LONGI_ACC_MIN, veh.LONGI_ACC_MAX)
            veh.action['acc_y'] = np.clip(veh.action['acc_y'], veh.LAT_ACC_MIN, veh.LAT_ACC_MAX)

    def RSS_algorithm(self, egoVehicle, objectVehicle, output_lon_lat_safe_flag=False):
        """
        Return restrictions of each one vehicle in the pair.
        """
        pair_id = (egoVehicle.id, objectVehicle.id)

        # Add the vehicle pair if the first time occur.
        if pair_id not in self.RSS_state_dict:
            self.RSS_state_dict[pair_id] = {'dangerousTimeLon': None, 'dangerousTimeLat': None}

        # Pre-check using sphere
        precheck_safe = self._precheck_sphere_safe(egoVehicle, objectVehicle)
        if precheck_safe:
            lon_safe, lat_safe = True, True
            ego_restrictions = RSS_restrictions([], [], [], [])  # No restrictions are needed for c1.
            obj_restrictions = RSS_restrictions([], [], [], [])  # No restrictions are needed for c2.
            if output_lon_lat_safe_flag:
                return ego_restrictions, obj_restrictions, lon_safe, lat_safe
            else:
                return ego_restrictions, obj_restrictions

        # Longitudinal and Lateral check
        ego_is_front, closest_longi_dist, ego_is_upper, closest_lat_dist = self._closest_longitudinal_and_lateral_distance(ego=egoVehicle, other=objectVehicle)
        lon_safe = self._longitudinal_safety_check(egoVehicle=egoVehicle, objectVehicle=objectVehicle, ego_is_front=ego_is_front, closest_longi_dist=closest_longi_dist)
        lat_safe = self._lateral_safety_check(egoVehicle=egoVehicle, objectVehicle=objectVehicle, ego_is_upper=ego_is_upper, closest_lat_dist=closest_lat_dist, closest_longi_dist=closest_longi_dist)
        ego_restrictions, obj_restrictions = self._proper_response_generate(egoVehicle=egoVehicle, objectVehicle=objectVehicle, lon_safe=lon_safe,
                                                                            lat_safe=lat_safe, ego_is_front=ego_is_front, ego_is_upper=ego_is_upper)
        if (not lon_safe) and (not lat_safe):
            egoVehicle.RSS_dangerous, objectVehicle.RSS_dangerous = True, True

        if output_lon_lat_safe_flag:
            return ego_restrictions, obj_restrictions, lon_safe, lat_safe
        else:
            return ego_restrictions, obj_restrictions

    def _lon_proper_response(self, ego_is_front):
        """
        Definition 4.
        """
        if not ego_is_front:  # ego is behind (c1 in the paper), other is front (c2)
            ego_lon_restriction_sign = ['leq']
            ego_lon_restriction_value = [-self.VehicleRssDynamics.alphaLon.brakeMin]
            obj_lon_restriction_sign = ['geq']
            obj_lon_restriction_value = [-self.VehicleRssDynamics.alphaLon.brakeMax]
        else:
            obj_lon_restriction_sign = ['leq']
            obj_lon_restriction_value = [-self.VehicleRssDynamics.alphaLon.brakeMin]
            ego_lon_restriction_sign = ['geq']
            ego_lon_restriction_value = [-self.VehicleRssDynamics.alphaLon.brakeMax]

        return ego_lon_restriction_sign, ego_lon_restriction_value, obj_lon_restriction_sign, obj_lon_restriction_value

    def _lat_proper_response(self, egoVehicle, objectVehicle, ego_is_upper):
        if ego_is_upper:
            obj_is_upper = False
        else:
            obj_is_upper = True
        ego_lat_restriction_sign, ego_lat_restriction_value = self._lat_proper_response_one_vehicle(veh=egoVehicle, veh_is_upper=ego_is_upper)
        obj_lat_restriction_sign, obj_lat_restriction_value = self._lat_proper_response_one_vehicle(veh=objectVehicle, veh_is_upper=obj_is_upper)

        return ego_lat_restriction_sign, ego_lat_restriction_value, obj_lat_restriction_sign, obj_lat_restriction_value

    def _lat_proper_response_one_vehicle(self, veh, veh_is_upper):
        veh_v_y = veh.speed_lat
        lat_restriction_sign, lat_restriction_value = [], []
        if veh_is_upper:
            if veh_v_y > 0:
                lat_restriction_sign = ['geq']
                lat_restriction_value = [-self.VehicleRssDynamics.alphaLat.brakeMin]
            if veh_v_y == 0:
                lat_restriction_sign = ['geq']
                lat_restriction_value = [0.]
            if veh_v_y < 0:
                lat_restriction_sign = ['geq']
                lat_restriction_value = [-veh_v_y / 0.1]
        if not veh_is_upper:
            if veh_v_y < 0:
                lat_restriction_sign = ['leq']
                lat_restriction_value = [self.VehicleRssDynamics.alphaLat.brakeMin]
            if veh_v_y == 0:
                lat_restriction_sign = ['leq']
                lat_restriction_value = [0.]
            if veh_v_y > 0:
                lat_restriction_sign = ['leq']
                lat_restriction_value = [-veh_v_y / 0.1]
        return lat_restriction_sign, lat_restriction_value

    def _proper_response_generate(self, egoVehicle, objectVehicle, lon_safe, lat_safe, ego_is_front, ego_is_upper):
        """
        Definition 9. in RSS.
        """
        response_needed = (not lon_safe) and (not lat_safe)
        if not response_needed:
            ego_restrictions = RSS_restrictions([], [], [], [])  # No restrictions are needed for c1.
            obj_restrictions = RSS_restrictions([], [], [], [])  # No restrictions are needed for c2.
            return ego_restrictions, obj_restrictions

        else:
            pair_id = self._pair_id_tuple(egoVehicle, objectVehicle)

            dangerousTimeLon = self.RSS_state_dict[pair_id]['dangerousTimeLon']
            dangerousTimeLat = self.RSS_state_dict[pair_id]['dangerousTimeLat']

            ego_lon_restriction_sign, ego_lon_restriction_value, obj_lon_restriction_sign, obj_lon_restriction_value = self._lon_proper_response(
                ego_is_front=ego_is_front)  # Lon proper response (Definition 4.)
            ego_lat_restriction_sign, ego_lat_restriction_value, obj_lat_restriction_sign, obj_lat_restriction_value = self._lat_proper_response(egoVehicle=egoVehicle,
                                                                                                                                                 objectVehicle=objectVehicle,
                                                                                                                                                 ego_is_upper=ego_is_upper)  #
            # Lat proper response (Definition 8.)

            if dangerousTimeLon > dangerousTimeLat:  # Perform Lon proper response only
                if egoVehicle.lane_index != objectVehicle.lane_index and abs(egoVehicle.position[1]-objectVehicle.position[1]) > 2.5:
                    ego_lat_restriction_sign, ego_lat_restriction_value, obj_lat_restriction_sign, obj_lat_restriction_value = [], [], [], []
                else:
                    pass
            elif dangerousTimeLon < dangerousTimeLat:  # Perform Lat proper response only
                if egoVehicle.lane_index != objectVehicle.lane_index and abs(egoVehicle.position[1]-objectVehicle.position[1]) > 2.5: #! not same lane or lateral distance > 2
                    ego_lon_restriction_sign, ego_lon_restriction_value, obj_lon_restriction_sign, obj_lon_restriction_value = [], [], [], []
                else:
                    pass
            else:  # Both Lon and Lat response. Then nothing needs to be changed
                pass

            # Add some rule based restrictions here
            overlap_lateral_dist_threshold = 2.0
            pred_time = 1.0 # s
            ego_is_front, closest_longi_dist, ego_is_upper, closest_lat_dist = self._closest_longitudinal_and_lateral_distance(ego=egoVehicle, other=objectVehicle)
            if egoVehicle.position[0] > objectVehicle.position[0]:
                rangerate = egoVehicle.speed_long-objectVehicle.speed_long
            else:
                rangerate = objectVehicle.speed_long-egoVehicle.speed_long
            predict_longitudinal_dist_change = rangerate*pred_time
            if closest_lat_dist < 2: # one of the vehicle is not in the center of the road
                if closest_longi_dist < 0 or closest_longi_dist + predict_longitudinal_dist_change < 0: # Lonitudinal overlap
                    if ego_is_upper:
                        if egoVehicle.speed_lat <= 0:
                            ego_lat_restriction_sign.append("geq")
                            ego_lat_restriction_value.append(0)
                        else:
                            ego_lat_restriction_sign.append("geq")
                            ego_lat_restriction_value.append(-egoVehicle.speed_lat/0.1)
                    else:
                        if egoVehicle.speed_lat >= 0:
                            ego_lat_restriction_sign.append("leq")
                            ego_lat_restriction_value.append(0)
                        else:
                            ego_lat_restriction_sign.append("leq")
                            ego_lat_restriction_value.append(-egoVehicle.speed_lat/0.1)

            # Restrictions for both vehicles.
            ego_restrictions = RSS_restrictions(ego_lon_restriction_sign, ego_lon_restriction_value, ego_lat_restriction_sign, ego_lat_restriction_value)
            obj_restrictions = RSS_restrictions(obj_lon_restriction_sign, obj_lon_restriction_value, obj_lat_restriction_sign, obj_lat_restriction_value)
        return ego_restrictions, obj_restrictions

    @staticmethod
    def _cal_safe_dis_lat(v_upper, v_lower, rho, mu, a_accel_max_lat, a_brake_min_lat):
        """
        Definition 6. in RSS.
        # v is positive when the direction is down, is positive when the direction is up
        v is positive when the direction is up, is negative when the direction is down
        Both vehicles using a_accel_max_lat to accelerate during rho and using at least a_brake_min_lat to avoid
        All acc here are absolute value
        """
        v_upper_rho, v_lower_rho = v_upper - a_accel_max_lat * rho, v_lower + a_accel_max_lat * rho

        d_upper = -(v_upper + v_upper_rho) * rho / 2 + (v_upper_rho ** 2) / (2 * a_brake_min_lat) if v_upper_rho <= 0 \
            else -(v_upper + v_upper_rho) * rho / 2 - (v_upper_rho ** 2) / (2 * a_brake_min_lat)
        d_lower = (v_lower + v_lower_rho) * rho / 2 + (v_lower_rho ** 2) / (2 * a_brake_min_lat) if v_lower_rho >= 0 \
            else (v_lower + v_lower_rho) * rho / 2 - (v_lower_rho ** 2) / (2 * a_brake_min_lat)

        d = d_upper + d_lower
        d_min_lat = mu + np.max([0, d])

        return d_min_lat

    # @profile
    def RSS_act_CAV(self, env, action = None, longitudinal_speed = None, lateral_speed = None):
        """
        Generate restrictions for each vehicle in the environment.
        """
        self.current_time = env.simulator.sumo_time_stamp  # Update current time
        self.RSS_restrictions_dict = {
            "CAV":{"lon_sign":[],"lon_val":[],"lat_sign":[],"lat_val":[]}
        }  # Reset the dict
        cav = env.vehicle_list["CAV"]
        cav_info = cav.observation.local["CAV"]
        if lateral_speed is not None:
            cav_info[50] = lateral_speed
        else:
            cav_info[50] = env.vehicle_list["CAV"].lateral_speed
        if longitudinal_speed is not None:
            cav_info[64] = longitudinal_speed
        # self.cav_veh.update(cav_info, action, cav.controller.highly_critical)
        self.cav_veh.update(cav_info, action)
        cav_context_info = cav.observation.context

        # Loop to examine any pair of vehicles.
        for veh_id in cav_context_info.keys():
            bv_veh = RSS_vehicle(veh_id)
            bv_veh.update(cav_context_info[veh_id])
            ego_restrictions, _ = self.RSS_algorithm(egoVehicle=self.cav_veh, objectVehicle=bv_veh)
            
            # Update the restriction dict
            self._add_vehicle_restrictions_to_dict(veh=self.cav_veh, veh_restrictions=ego_restrictions)
        
        return self.RSS_restrictions_dict

    def RSS_step_CAV(self, env):
        """
        Apply the restrictions on vehicle actions.
        """
        veh_id = self.cav_veh.id
        veh_RSS_restrictions = self.RSS_restrictions_dict[veh_id]
        acc_x_after_RSS_regulation = self._execute_RSS_restriction(action=self.cav_veh.action['acc_x'], restriction_sign=veh_RSS_restrictions['lon_sign'],
                                                                restriction_value=veh_RSS_restrictions['lon_val'])
        acc_y_after_RSS_regulation = self._execute_RSS_restriction(action=self.cav_veh.action['acc_y'], restriction_sign=veh_RSS_restrictions['lat_sign'],
                                                                restriction_value=veh_RSS_restrictions['lat_val'])

        self.cav_veh.RSS_action['acc_x'] = acc_x_after_RSS_regulation
        self.cav_veh.RSS_action['acc_y'] = acc_y_after_RSS_regulation

        # Update whether the action has been modified by RSS restrictions.
        rand_number = random.uniform(0, 1)
        self.cav_veh.RSS_randomly_choose_not_control = False
        if ((acc_x_after_RSS_regulation != self.cav_veh.action['acc_x']) or (acc_y_after_RSS_regulation != self.cav_veh.action['acc_y'])):
            if rand_number > self.RSS_not_activate_prob:
                self.cav_veh.RSS_control = True
                self.cav_veh.action['acc_x'] = acc_x_after_RSS_regulation
                self.cav_veh.action['acc_y'] = acc_y_after_RSS_regulation
                # Cannot drive backwards
                veh_v_x = self.cav_veh.speed_long
                if (veh_v_x <= 0) or (veh_v_x + self.cav_veh.action['acc_x'] * 0.1) <= 0:
                    self.cav_veh.action['acc_x'] = np.clip(self.cav_veh.action['acc_x'], 0., np.inf)

                # Physical constraints
                self.cav_veh.action['acc_x'] = np.clip(self.cav_veh.action['acc_x'], self.cav_veh.LONGI_ACC_MIN, self.cav_veh.LONGI_ACC_MAX)
                self.cav_veh.action['acc_y'] = np.clip(self.cav_veh.action['acc_y'], self.cav_veh.LAT_ACC_MIN, self.cav_veh.LAT_ACC_MAX)
            else:  # There are some probability that RSS not activate at the moment.
                self.cav_veh.RSS_control = False
                self.cav_veh.RSS_randomly_choose_not_control = True
        else:
            self.cav_veh.RSS_control = False

        return self.cav_veh.action

    def update_states(self, states_dict, init_time_step):
        self.RSS_state_dict = {}
        for bv_id in states_dict:
            pair = ("CAV", bv_id)
            newstates = {}
            for key in states_dict[bv_id]:
                if states_dict[bv_id][key] is not None:
                    newstates[key] = round(states_dict[bv_id][key]-round(float(init_time_step),1),1)
                else:
                    newstates[key] = None
            self.RSS_state_dict[pair] = newstates