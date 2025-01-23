import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork, LaneletType
from commonroad_rl.gym_commonroad.action import Vehicle
from commonroad_rl.gym_commonroad.observation import ObservationCollector
from commonroad_rl.gym_commonroad.utils.scenario import approx_orientation_vector
from commonroad.geometry.shape import Shape
from commonroad.scenario.scenario import Scenario
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from commonroad.scenario.obstacle import ObstacleType, Obstacle, DynamicObstacle
from commonroad.prediction.prediction import Occupancy
import commonroad_dc.pycrcc as pycrcc


def check_collision_type(info, ego_vehicle: Vehicle, observation_collector,
                         scenario: Scenario, episode, lane_change_time_threshold, local_ccosy):
    other_vehicle_at_fault, reason = _current_collision_type_checking(
        ego_vehicle, observation_collector, scenario, lane_change_time_threshold, local_ccosy)
    #if other_vehicle_at_fault:
    #    print(f"Unexpected Collision not caused by ego vehicle in {episode} with reason: {reason}")

    info['valid_collision'] = not other_vehicle_at_fault
    info['collision_reason'] = reason
    return info

def _current_collision_type_checking(ego_vehicle: Vehicle, observation_collector,
                                     scenario: Scenario, lane_change_time_threshold, local_ccosy):

    # Determine collision participants
    collision_ego_vehicle = ego_vehicle.collision_object
    collision_objects = observation_collector.get_collision_checker().find_all_colliding_objects(collision_ego_vehicle)
    current_step = ego_vehicle.current_time_step

    # TODO: refactor once issue https://gitlab.lrz.de/cps/commonroad-drivability-checker/-/issues/17 is closed
    # get positions of all obstacles in the scenario
    obstacle_state_dict = dict()
    for obstacle in scenario.obstacles:
        obstacle_state = obstacle.state_at_time(current_step)
        if obstacle_state is not None:
            obstacle_state_dict[obstacle] = obstacle_state

    collision_obstacles = {}
    for collision_object in collision_objects:
        collision_shape = collision_object.obstacle_at_time(ego_vehicle.current_time_step)
        if collision_shape.collide(observation_collector.road_edge["boundary_collision_object"]):
            return True, "Other vehicle was outside of lanelet network"
        for obstacle, obstacle_state in obstacle_state_dict.items():
            if np.allclose(collision_shape.center(), obstacle_state.position):
                collision_obstacles[obstacle] = obstacle_state

    ego_shape = Rectangle(length=ego_vehicle.parameters.l,
                          width=ego_vehicle.parameters.w,
                          center=ego_vehicle.state.position,
                          orientation=ego_vehicle.state.orientation)
    ego_position = ego_vehicle.state.position

    # TODO: save lane results in pre-processing
    merge_results = create_lanes(scenario.lanelet_network, merging_length=1000.)
    lanes = [merge_result[1] for merge_result in merge_results]
    time_steps_back = int(lane_change_time_threshold / scenario.dt)

    for obstacle, obstacle_state in collision_obstacles.items():
        # get edges of obstacle shape and check if it was fully on the lanelet network
        if current_step == obstacle.initial_state.time_step:
            return True, "Other vehicle suddenly appeared"
        if obstacle.obstacle_type == ObstacleType.CAR:
            # compute contact point of collision
            contact_point = _find_collision_point(ego_shape, obstacle.occupancy_at_time(current_step))
            lanelets_contact_point = set(scenario.lanelet_network.find_lanelet_by_position([np.array(contact_point)])[0])

            def find_lanelet_id_for_state(observation_collector, state, scenario):
                lanelet_polygons, lanelet_polygons_sg = observation_collector._get_lanelet_polygons(str(scenario.scenario_id))
                lanelet_ids = observation_collector.sorted_lanelets_by_state(
                    scenario, state, lanelet_polygons, lanelet_polygons_sg
                )
                return lanelet_ids[0]

            def find_lanelet_ids(state_list, time_steps):
                if len(state_list) >= time_steps:
                    state_list = state_list[-time_steps:]
                return set([find_lanelet_id_for_state(observation_collector, state, scenario)
                                      for state in state_list])

            def check_lane_change(lanelet_ids, lanes):
                for lane in lanes:
                    if (lanelet_ids.issubset(set(lane))):
                        return False
                return True

            obstacle_state_list = [obstacle.state_at_time(time_step) for time_step in
                                   range(obstacle.initial_state.time_step, current_step + 1)]

            ego_lanelet_ids = find_lanelet_ids(ego_vehicle.state_list, time_steps_back)
            obstacle_lanelet_ids = find_lanelet_ids(obstacle_state_list, time_steps_back)
            obstacle_exceeds_lane = True
            ego_exceeds_lane = True
            for lanelet_contact_point in lanelets_contact_point:
                if lanelet_contact_point in obstacle_lanelet_ids:
                    obstacle_exceeds_lane = False
                if lanelet_contact_point in ego_lanelet_ids:
                    ego_exceeds_lane = False
            if obstacle_exceeds_lane:
                return True, "Other vehicle drove out of its lane"
            if ego_exceeds_lane:
                return False, "Ego vehicle drove out of its lane"

            # if len(contact_point) == 0:
            #     # Very rare (possibly impossible) event, that none of the vehicle's corners are collision contact point
            #     return False, "Collision reason unclear"
            #

            #
            # # Calculate possible lane path based on current lanelet of the obstacle
            # laneset_obst = _calculate_lanelet_set(obstacle_state.position, scenario.lanelet_network, ego_shape.length)
            #
            # # Collision contact point on different lanelet as other vehicle's center? -> Other vehicle drove out of lane
            # if len(set.intersection(lanelet_contact_point, laneset_obst)) == 0:
            #     return True, "Other vehicle drove out of its lane"
            #
            # laneset_ego = _calculate_lanelet_set(ego_position, scenario.lanelet_network, ego_shape.length)
            #
            # # Collision contact point on different lanelet as ego vehicle's center? -> Ego vehicle drove out of lane
            # if len(set.intersection(lanelet_contact_point, laneset_ego)) == 0:
            #     return False, "Ego vehicle drove out of its lane"

            # determine back collision
            s_ego, _ = local_ccosy.convert_to_curvilinear_coords(ego_position[0], ego_position[1])
            s_obs, _ = local_ccosy.convert_to_curvilinear_coords(obstacle_state.position[0], obstacle_state.position[1])
            rear_collision = s_obs <= s_ego

            if rear_collision:
                # check if ego switched lane
                if check_lane_change(ego_lanelet_ids, lanes):
                    return False, "Ego cut in"
                else:
                    return True, "Other collides from rear"
            else:
                # check if other switched lane
                if check_lane_change(obstacle_lanelet_ids, lanes):
                    return True, "Other cut in"
                else:
                    return False, "Ego collides from rear"

            # previous_position = None
            # # Determine back collision
            # ego_vector = approx_orientation_vector(ego_vehicle.state.orientation)
            # rear_collision = False
            #
            # distance_vector = obstacle.obstacle_shape.center() - ego_position
            # distance = np.linalg.norm(distance_vector)
            # distance_vector /= distance
            # delta_angle = np.arccos(np.vdot(ego_vector, distance_vector))
            # if delta_angle < 5 / 6 * np.pi:  # 150 degree
            #     rear_collision = True

            # if rear_collision:
            #
            #     # Check if ego vehicle did not stay in lane/changed lane? Yes -> Ego vehicle's fault
            #
            #     distance = 0
            #     if len(ego_vehicle.state_list) > time_steps_back + 1:
            #         previous_position = ego_vehicle.state_list[-(time_steps_back + 1)].position
            #
            #         for i in range((time_steps_back - 1) // 5):
            #             distance += np.abs(np.linalg.norm(
            #                 ego_vehicle.state_list[-1 - (5 * i)].position - ego_vehicle.state_list[
            #                     -1 - (5 * (i + 1))].position))
            #
            #         if (time_steps_back - 1) % 5 != 0:
            #             distance += np.abs(np.linalg.norm(ego_vehicle.state_list[-1 - (
            #                         5 * ((time_steps_back - 1) // 5))].position - previous_position))
            #     else:
            #         previous_position = ego_vehicle.initial_state.position
            #         for i in range((len(ego_vehicle.state_list) - 1) // 5):
            #             distance += np.abs(np.linalg.norm(
            #                 ego_vehicle.state_list[-1 - (5 * i)].position - ego_vehicle.state_list[
            #                     -1 - (5 * (i + 1))].position))
            #
            #         if (len(ego_vehicle.state_list) - 1) % 5 != 0:
            #             distance += np.abs(np.linalg.norm(ego_vehicle.state_list[-1 - (
            #                         5 * ((time_steps_back - 1) // 5))].position - previous_position))
            #
            #     preceding_lanes = _calculate_preceding_lanelet_set(ego_position, scenario.lanelet_network, distance)
            #     previous_lanes = scenario.lanelet_network.find_lanelet_by_position([previous_position])[0]
            #     for lane in previous_lanes:
            #         if not (lane in preceding_lanes):
            #             return False, "Ego vehicle joined the current lane in an unsafe manner"
            #
            #     return True, "Other vehicle drove into back of ego vehicle"
            #
            # # Check if other vehicle did not stay in lane/changed lane? Yes -> Other vehicle's fault
            #
            # previous_position = obstacle.occupancy_at_time(max(current_step - time_steps_back, 0)).shape.center
            # distance = 0
            # step_limit = np.abs(current_step - time_steps_back)
            # for i in range((step_limit - 1) // 5):
            #     distance += np.abs(
            #         np.linalg.norm(obstacle.occupancy_at_time(max(current_step - (i * 5), 0)).shape.center
            #                        - obstacle.occupancy_at_time(max(current_step - ((i + 1) * 5), 0)).shape.center))
            #
            # if (step_limit - 1) % 5 != 0:
            #     distance += np.abs(np.linalg.norm(obstacle.occupancy_at_time(
            #         max(current_step - ((step_limit - 1 // 5) * 5),
            #             0)).shape.center - obstacle._initial_occupancy_shape.center))
            #
            # preceding_lanes = _calculate_preceding_lanelet_set(obstacle_state.position, scenario.lanelet_network,
            #                                                    distance)
            # previous_lanes = scenario.lanelet_network.find_lanelet_by_position([previous_position])[0]
            # for lane in previous_lanes:
            #     if not (lane in preceding_lanes):
            #         return True, "Other vehicle joined the current lane in an unsafe manner"
            #
            # return False, "Ego vehicle drove into other vehicle"

    return False, "Unsuccessful determination"


def create_lanes(lanelet_network: LaneletNetwork, merging_length: float = 600.):
    """
    Creates lanes for road network

    :param merging_length: length for merging successors
    :param lanelet_network:
    :return:
    """
    lane_lanelets = []
    start_lanelets = []

    for lanelet in lanelet_network.lanelets:
        if len(lanelet.predecessor) == 0:
            start_lanelets.append(lanelet)
        else:
            predecessors = [lanelet_network.find_lanelet_by_id(pred_id) for pred_id in lanelet.predecessor]
            for pred in predecessors:
                if not lanelet.lanelet_type == pred.lanelet_type:
                    start_lanelets.append(lanelet)

    for lanelet in start_lanelets:
        # if LaneletType.ACCESS_RAMP in lanelet.lanelet_type:
        #     lanelet_type = LaneletType.ACCESS_RAMP
        # elif LaneletType.EXIT_RAMP in lanelet.lanelet_type:
        #     lanelet_type = LaneletType.EXIT_RAMP
        # elif LaneletType.MAIN_CARRIAGE_WAY in lanelet.lanelet_type:
        #     lanelet_type = LaneletType.MAIN_CARRIAGE_WAY
        # else:
        #     lanelet_type = None

        merged_lanelets, merge_jobs = \
            Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                lanelet, lanelet_network, merging_length)
        if len(merged_lanelets) == 0 or len(merge_jobs) == 0:
            merged_lanelets.append(lanelet)
            merge_jobs.append([lanelet.lanelet_id])
        for idx in range(len(merged_lanelets)):
            lane_lanelets.append((merged_lanelets[idx], merge_jobs[idx]))

    return lane_lanelets


def _find_collision_point(ego_shape: Shape, obstacle_occupancy: Occupancy):
    obstacle_vertices = obstacle_occupancy.shape.vertices

    # collision obstacle's corners in ego shape?
    for obstacle_vertice in obstacle_vertices:
        if ego_shape.contains_point(obstacle_vertice):
            return obstacle_vertice

    # ego's corner in obstacle's shape?
    # ego corners in collision obstacle's shape?
    for ego_vertice in ego_shape.vertices:
        if obstacle_occupancy.shape.contains_point(ego_vertice):
            return ego_vertice

    raise ValueError


# def _find_lanelet_predecessors_in_range(lane: Lanelet, lanelet_network: "LaneletNetwork", max_length=50.0) -> [[int]]:
#     """
#     Finds all possible predecessors paths (id sequences) within max_length. Adapted from Commonroad-IO.
#
#     :param lanelet_network: lanelet network
#     :param max_length: abort once length of path is reached
#     :return: list of lanelet IDs
#     """
#     paths = [[s] for s in lane.predecessor]
#     paths_final = []
#     lengths = [lanelet_network.find_lanelet_by_id(s).distance[-1] for s in lane.predecessor]
#     while paths:
#         paths_next = []
#         lengths_next = []
#         for p, le in zip(paths, lengths):
#             predecessors = lanelet_network.find_lanelet_by_id(p[-1]).predecessor
#             if not predecessors:
#                 paths_final.append(p)
#             else:
#                 for s in predecessors:
#                     if s in p or s == lane.lanelet_id or le >= max_length:
#                         # prevent loops and consider length of first predecessor
#                         paths_final.append(p)
#                         continue
#
#                     l_next = le + lanelet_network.find_lanelet_by_id(s).distance[-1]
#                     if l_next < max_length:
#                         paths_next.append(p + [s])
#                         lengths_next.append(l_next)
#                     else:
#                         paths_final.append(p + [s])
#
#         paths = paths_next
#         lengths = lengths_next
#
#     return paths_final
#
#
# def _calculate_lanelet_set(position, lanelet_network, max_length):
#     laneset = set()
#     for lanelet in lanelet_network.find_lanelet_by_position([position])[0]:
#         laneset.add(lanelet)
#         for i in lanelet_network.find_lanelet_by_id(lanelet).find_lanelet_successors_in_range(lanelet_network,
#                                                                                               max_length=max_length):
#             laneset = laneset.union(set(i))
#         for i in _find_lanelet_predecessors_in_range(lanelet_network.find_lanelet_by_id(lanelet), lanelet_network,
#                                                      max_length):
#             laneset = laneset.union(set(i))
#
#     return laneset
#
#
# def _calculate_preceding_lanelet_set(position, lanelet_network, max_length):
#     laneset = set()
#     for lanelet in lanelet_network.find_lanelet_by_position([position])[0]:
#         laneset.add(lanelet)
#         for i in _find_lanelet_predecessors_in_range(lanelet_network.find_lanelet_by_id(lanelet), lanelet_network,
#                                                      max_length):
#             laneset = laneset.union(set(i))
#
#     return laneset
#
#
# def _calculate_preceding_adjacent_lanes(position, lanelet_network, max_length):
#     laneset = set()
#     adj_set = set()
#
#     for lanelet in lanelet_network.find_lanelet_by_position([position])[0]:
#         laneset.add(lanelet)
#         for path in _find_lanelet_predecessors_in_range(lanelet, lanelet_network, max_length):
#             laneset = laneset.union(set(path))
#
#     for lane in laneset:
#         if lane.adj_left is not None:
#             if lane.adj_left != lanelet and lane.adj_left_same_direction:
#                 adj_set.add(lane.adj_left)
#         if lane.adj_right is not None:
#             if lane.adj_right != lanelet and lane.adj_right_same_direction:
#                 adj_set.add(lane.adj_right)
#
#     return adj_set
