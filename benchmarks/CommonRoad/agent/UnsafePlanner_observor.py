import math

class Unsafeplanner_observor:
    def __init__(self):
        pass

    def observation_from_sumo(self, simulator):
        """
        Get observation from SUMO simulator.
        """
        cav = simulator.env.vehicle_list["CAV"]
        cav_info = cav.observation.local["CAV"]
        last_cav_info = cav._last_observation.local["CAV"]
        cav_context = cav.observation.context_raw
        cav_pos = cav_info[66]
        last_cav_pos = last_cav_info[66]
        cav_heading = (90 - cav_info[67]) * math.pi / 180
        last_cav_heading = (90 - last_cav_info[67]) * math.pi / 180
        cav_velocity = cav_info[64]
        cav_acceleration = cav_info[114]
        cav_yaw_rate = (cav_heading - last_cav_heading) / cav.step_size

        observation = []
        # distance_goal_long
        observation.append(800 - cav_pos[0] + math.cos(cav_heading)*5/2)
        # distance_goal_long_advance
        if simulator.current_time_steps == 1:
            observation.append(0.0)
        else:
            observation.append(cav_pos[0] - math.cos(cav_heading)*5/2 - last_cav_pos[0] + math.cos(last_cav_heading)*5/2)
        # distance_goal_lat
        observation.append(0.0)
        # distance_goal_lat_advance
        observation.append(0.0)
        # distance_goal_time
        observation.append(0.0)
        # distance_goal_orientation
        if cav_heading < -0.2:
            observation.append(cav_heading - (-0.2))
        elif cav_heading > 0.2:
            observation.append(cav_heading - 0.2)
        else:
            observation.append(0.0)
        # distance_goal_velocity
        if cav_velocity < 20:
            observation.append(cav_velocity - 20)
        elif cav_velocity > 45:
            observation.append(cav_velocity - 45)
        else:
            observation.append(0.0)
        # is_goal_reached
        observation.append(int(cav_pos[0]-math.cos(cav_heading)*5/2 >= 800) + 0.0)
        # is_time_out
        observation.append(int(simulator.current_time_steps >= 200) + 0.0)
        # v_ego
        observation.append(cav_velocity)
        # a_ego
        observation.append(cav_acceleration)
        # relative_heading
        observation.append(cav_heading)
        # global_turn_rate
        observation.append(cav_yaw_rate)
        # is_friction_violation
        observation.append(0.0)
        # remaining_steps
        observation.append(200.0 - simulator.current_time_steps)

        # get surrounding vehicles
        if cav_pos[1]-math.sin(cav_heading)*5/2 < 44:
            same_lane_range = [40,44]
        elif cav_pos[1]-math.sin(cav_heading)*5/2 < 48:
            same_lane_range = [44,48]
        else:
            same_lane_range = [48,52]
        left_lane_range = [same_lane_range[0]+4, same_lane_range[1]+4]
        right_lane_range = [same_lane_range[0]-4, same_lane_range[1]-4]
        left_lead_veh_list = []
        same_lead_veh_list = []
        right_lead_veh_list = []
        left_follow_veh_list = []
        same_follow_veh_list = []
        right_follow_veh_list = []
        for veh_id in cav_context.keys():
            bv_pos = cav_context[veh_id][66]
            bv_heading = (90 - cav_context[veh_id][67]) * math.pi / 180
            rel_dis_long = bv_pos[0]-math.cos(bv_heading)*5/2-cav_pos[0]+math.cos(cav_heading)*5/2
            rel_dis_lat = bv_pos[1]-math.sin(bv_heading)*5/2-cav_pos[1]+math.sin(cav_heading)*5/2
            bv_velocity = math.sqrt(cav_context[veh_id][64]**2 + cav_context[veh_id][50]**2)
            if rel_dis_long < 0:
                rel_v_long = cav_velocity - bv_velocity * math.cos(cav_heading-bv_heading)
            else:
                rel_v_long = bv_velocity * math.cos(bv_heading-cav_heading) - cav_velocity
            if math.sqrt(rel_dis_long**2 + rel_dis_lat**2) > 100:
                continue
            if bv_pos[1] - math.sin(bv_heading)*5/2 < same_lane_range[1] and bv_pos[1] - math.sin(bv_heading) * 5/2 > same_lane_range[0] and rel_dis_long > 0:
                same_lead_veh_list.append([rel_dis_long, rel_v_long])
            elif bv_pos[1] - math.sin(bv_heading)*5/2 < same_lane_range[1] and bv_pos[1] - math.sin(bv_heading)*5/2 > same_lane_range[0] and rel_dis_long < 0:
                same_follow_veh_list.append([rel_dis_long, rel_v_long])
            elif bv_pos[1] - math.sin(bv_heading)*5/2 < left_lane_range[1] and bv_pos[1] - math.sin(bv_heading)*5/2 > left_lane_range[0] and rel_dis_long > 0:
                left_lead_veh_list.append([rel_dis_long, rel_v_long])
            elif bv_pos[1] - math.sin(bv_heading)*5/2 < left_lane_range[1] and bv_pos[1] - math.sin(bv_heading)*5/2 > left_lane_range[0] and rel_dis_long < 0:
                left_follow_veh_list.append([rel_dis_long, rel_v_long])
            elif bv_pos[1] - math.sin(bv_heading)*5/2 < right_lane_range[1] and bv_pos[1] - math.sin(bv_heading)*5/2 > right_lane_range[0] and rel_dis_long > 0:
                right_lead_veh_list.append([rel_dis_long, rel_v_long])
            elif bv_pos[1] - math.sin(bv_heading)*5/2 < right_lane_range[1] and bv_pos[1] - math.sin(bv_heading)*5/2 > right_lane_range[0] and rel_dis_long < 0:
                right_follow_veh_list.append([rel_dis_long, rel_v_long]) 
        left_lead_veh_list.sort(key=lambda x: x[0])
        same_lead_veh_list.sort(key=lambda x: x[0])
        right_lead_veh_list.sort(key=lambda x: x[0])
        left_follow_veh_list.sort(key=lambda x: x[0])
        same_follow_veh_list.sort(key=lambda x: x[0])
        right_follow_veh_list.sort(key=lambda x: x[0])
        # v_rel_left_follow
        if len(left_follow_veh_list) == 0:
            observation.append(0.0)
        else:
            observation.append(left_follow_veh_list[-1][1])
        # v_rel_same_follow
        if len(same_follow_veh_list) == 0:
            observation.append(0.0)
        else:
            observation.append(same_follow_veh_list[-1][1])
        # v_rel_right_follow
        if len(right_follow_veh_list) == 0:
            observation.append(0.0)
        else:
            observation.append(right_follow_veh_list[-1][1])
        # v_rel_left_lead
        if len(left_lead_veh_list) == 0:
            observation.append(0.0)
        else:
            observation.append(left_lead_veh_list[0][1])
        # v_rel_same_lead
        if len(same_lead_veh_list) == 0:
            observation.append(0.0)
        else:
            observation.append(same_lead_veh_list[0][1])
        # v_rel_right_lead
        if len(right_lead_veh_list) == 0:
            observation.append(0.0)
        else:
            observation.append(right_lead_veh_list[0][1])
        # p_rel_left_follow
        if len(left_follow_veh_list) == 0:
            observation.append(100.0)
        else:
            observation.append(abs(left_follow_veh_list[-1][0]))
        # p_rel_same_follow
        if len(same_follow_veh_list) == 0:
            observation.append(100.0)
        else:
            observation.append(abs(same_follow_veh_list[-1][0]))
        # p_rel_right_follow
        if len(right_follow_veh_list) == 0:
            observation.append(100.0)
        else:
            observation.append(abs(right_follow_veh_list[-1][0]))
        # p_rel_left_lead
        if len(left_lead_veh_list) == 0:
            observation.append(100.0)
        else:
            observation.append(abs(left_lead_veh_list[0][0]))
        # p_rel_same_lead
        if len(same_lead_veh_list) == 0:
            observation.append(100.0)
        else:
            observation.append(abs(same_lead_veh_list[0][0]))
        # p_rel_right_lead
        if len(right_lead_veh_list) == 0:
            observation.append(100.0)
        else:
            observation.append(abs(right_lead_veh_list[0][0]))
        # is_collision
        observation.append(0.0)
        # lane_change
        observation.append(0.0) # there is no lane change in cr scenario
        # is_off_road
        observation.append(0.0)
        # left_marker_distance
        observation.append(same_lane_range[1]-cav_pos[1]+math.sin(cav_heading)*5/2)
        # right_marker_distance
        observation.append(cav_pos[1]-same_lane_range[0]-math.sin(cav_heading)*5/2)
        # left_road_edge_distance
        observation.append(52-cav_pos[1]+math.sin(cav_heading)*5/2)
        # right_road_edge_distance
        observation.append(cav_pos[1]-40-math.sin(cav_heading)*5/2)
        # lat_offset
        observation.append(cav_pos[1]-sum(same_lane_range)/2-math.sin(cav_heading)*5/2)
        return observation