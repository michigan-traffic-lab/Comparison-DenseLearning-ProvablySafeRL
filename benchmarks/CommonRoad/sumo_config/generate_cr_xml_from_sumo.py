import xml.etree.ElementTree as ET
import os, math
from xml.dom import minidom

def generate_planning_problem_xml(output_path):
    '''generate planning problem'''

    planning_problem_dict = {
        'id': 1,
        'initialState': {
            'position': {'x': 400.0, 'y':46.0},
            #'position': {'x': 400.0, 'y':50.0},
            'orientation': 0.0,
            'time': 0,
            'velocity': 30.00,
            'acceleration': 0.0,
            'yawRate': 0.0,
            'slipAngle': 0.0
        },
        'goalRegion': {
            'position': {
                'length': 200.0,
                'width': 12.0,
                #'width': 4.0,
                'orientation': 0.0,
                'center': {'x': 900.0, 'y': 46.0}
            },
            'orientation': {'intervalStart': -0.2, 'intervalEnd': 0.2},
            'time': {'intervalStart': 0, 'intervalEnd': 200},
            'velocity': {'intervalStart': 20, 'intervalEnd': 45}
        }
    }

    root = ET.Element('planning_problems')
    
    planning_problem = ET.SubElement(root, 'planningProblem', id=str(planning_problem_dict['id']))
    initial_state = ET.SubElement(planning_problem, 'initialState')
    
    position = ET.SubElement(initial_state, 'position')
    point = ET.SubElement(position, 'point')
    ET.SubElement(point, 'x').text = str(planning_problem_dict['initialState']['position']['x'])
    ET.SubElement(point, 'y').text = str(planning_problem_dict['initialState']['position']['y'])
    
    orientation = ET.SubElement(initial_state, 'orientation')
    ET.SubElement(orientation, 'exact').text = str(planning_problem_dict['initialState']['orientation'])
    
    time = ET.SubElement(initial_state, 'time')
    ET.SubElement(time, 'exact').text = str(planning_problem_dict['initialState']['time'])
    
    velocity = ET.SubElement(initial_state, 'velocity')
    ET.SubElement(velocity, 'exact').text = str(planning_problem_dict['initialState']['velocity'])
    
    acceleration = ET.SubElement(initial_state, 'acceleration')
    ET.SubElement(acceleration, 'exact').text = str(planning_problem_dict['initialState']['acceleration'])
    
    yaw_rate = ET.SubElement(initial_state, 'yawRate')
    ET.SubElement(yaw_rate, 'exact').text = str(planning_problem_dict['initialState']['yawRate'])
    
    slip_angle = ET.SubElement(initial_state, 'slipAngle')
    ET.SubElement(slip_angle, 'exact').text = str(planning_problem_dict['initialState']['slipAngle'])
    
    goal_region = ET.SubElement(planning_problem, 'goalRegion')
    state = ET.SubElement(goal_region, 'state')
    
    position = ET.SubElement(state, 'position')
    rectangle = ET.SubElement(position, 'rectangle')
    ET.SubElement(rectangle, 'length').text = str(planning_problem_dict['goalRegion']['position']['length'])
    ET.SubElement(rectangle, 'width').text = str(planning_problem_dict['goalRegion']['position']['width'])
    ET.SubElement(rectangle, 'orientation').text = str(planning_problem_dict['goalRegion']['position']['orientation'])
    center = ET.SubElement(rectangle, 'center')
    ET.SubElement(center, 'x').text = str(planning_problem_dict['goalRegion']['position']['center']['x'])
    ET.SubElement(center, 'y').text = str(planning_problem_dict['goalRegion']['position']['center']['y'])
    
    orientation = ET.SubElement(state, 'orientation')
    ET.SubElement(orientation, 'intervalStart').text = str(planning_problem_dict['goalRegion']['orientation']['intervalStart'])
    ET.SubElement(orientation, 'intervalEnd').text = str(planning_problem_dict['goalRegion']['orientation']['intervalEnd'])
    
    time = ET.SubElement(state, 'time')
    ET.SubElement(time, 'intervalStart').text = str(planning_problem_dict['goalRegion']['time']['intervalStart'])
    ET.SubElement(time, 'intervalEnd').text = str(planning_problem_dict['goalRegion']['time']['intervalEnd'])
    
    velocity = ET.SubElement(state, 'velocity')
    ET.SubElement(velocity, 'intervalStart').text = str(planning_problem_dict['goalRegion']['velocity']['intervalStart'])
    ET.SubElement(velocity, 'intervalEnd').text = str(planning_problem_dict['goalRegion']['velocity']['intervalEnd'])
    
    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)

def generate_simple_initialization(output_path):
    '''generate simple SUMO initialization'''

    veh_dict = []
    veh_dict.append({'id':'veh1','type':'IDM','route':'route_0','lane':"0to1_0",'pos':420,'speed':15,'acc':0,'orientation':0})
    veh_dict.append({'id':'veh2','type':'IDM','route':'route_0','lane':"0to1_1",'pos':440,'speed':15,'acc':0,'orientation':0})
    veh_dict.append({'id':'veh3','type':'IDM','route':'route_0','lane':"0to1_2",'pos':420,'speed':15,'acc':0,'orientation':0})

    root = ET.Element('initialization')
    for veh in veh_dict:
        vehicle = ET.SubElement(root, 'vehicle', id=veh['id'])
        for keys in veh.keys():
            if keys == 'id':
                continue
            ET.SubElement(vehicle, keys).text = str(veh[keys])

    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    from xml.dom import minidom
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)

def parse_vehicle_types(rou_file_path):
    tree = ET.parse(rou_file_path)
    root = tree.getroot()
    vehicle_types = {}
    
    for vtype in root.findall('vType'):
        vtype_id = vtype.get('id')
        length = vtype.get('length')
        width = vtype.get('width')
        vehicle_types[vtype_id] = (length, width)
    
    return vehicle_types

def parse_sumo_map(sumo_map_path, interval):
    tree = ET.parse(sumo_map_path)
    root = tree.getroot()
    lanes = []
    
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            shape = lane.get('shape')
            width = float(lane.get('width'))
            p1, p2 = map(str, shape.split(' '))
            x1, y1 = map(float, p1.split(','))
            x2, y2 = map(float, p2.split(','))
            
            points = []
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            num_points = int(length // interval)
            for i in range(num_points):
                t = i / num_points
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                points.append((x, y))
            
            lanes.append((lane_id, points, width))
    
    return lanes

def calculate_normal_vector(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx**2 + dy**2)
    return (-dy / length, dx / length)

def position_along_lane_to_xy(lane_points, pos):
    total_length = 0
    for i in range(len(lane_points) - 1):
        p1 = lane_points[i]
        p2 = lane_points[i + 1]
        segment_length = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if total_length + segment_length >= pos:
            t = (pos - total_length) / segment_length
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            return x, y
        total_length += segment_length
    return lane_points[-1]

def generate_commonroad_map(lanes, output_path, initialization_path, planning_problem_path, rou_file_path):
    '''generate commonroad map'''
    root = ET.Element('commonRoad', commonRoadVersion="2017a", benchmarkID="1", timeStepSize="0.1", author="", affiliation="", source="", tags="", date="")

    lanes.sort(key=lambda lane: -sum(p[0]+p[1] for p in lane[1]))

    lane_id_to_points = {}
    for idx, (lane_id, points, width) in enumerate(lanes):
        lanelet = ET.SubElement(root, 'lanelet', id=str(idx + 1))
        left_bound = ET.SubElement(lanelet, 'leftBound')
        right_bound = ET.SubElement(lanelet, 'rightBound')
        lane_id_to_points[lane_id] = points

        half_width = width / 2
        
        for i in range(len(points)):
            if i == 0:
                normal = calculate_normal_vector(points[i], points[i + 1])
            elif i == len(points) - 1:
                normal = calculate_normal_vector(points[i - 1], points[i])
            else:
                normal1 = calculate_normal_vector(points[i - 1], points[i])
                normal2 = calculate_normal_vector(points[i], points[i + 1])
                normal = ((normal1[0] + normal2[0]) / 2, (normal1[1] + normal2[1]) / 2)
                length = math.sqrt(normal[0]**2 + normal[1]**2)
                normal = (normal[0] / length, normal[1] / length)
            
            left_point = ET.SubElement(left_bound, 'point')
            right_point = ET.SubElement(right_bound, 'point')
            
            left_x = f'{points[i][0] + normal[0] * half_width:.4f}'
            left_y = f'{points[i][1] + normal[1] * half_width:.4f}'
            right_x = f'{points[i][0] - normal[0] * half_width:.4f}'
            right_y = f'{points[i][1] - normal[1] * half_width:.4f}'

            ET.SubElement(left_point, 'x').text = left_x
            ET.SubElement(left_point, 'y').text = left_y
            ET.SubElement(right_point, 'x').text = right_x
            ET.SubElement(right_point, 'y').text = right_y

        if idx == 0:
            ET.SubElement(left_bound, 'lineMarking').text = "solid"
        else:
            ET.SubElement(left_bound, 'lineMarking').text = "dashed"
        if idx == len(lanes) - 1:
            ET.SubElement(right_bound, 'lineMarking').text = "solid"
        else:
            ET.SubElement(right_bound, 'lineMarking').text = "dashed"

        if idx > 0:
            ET.SubElement(lanelet, 'adjacentLeft', ref=str(idx), drivingDir="same")
        if idx < len(lanes) - 1:
            ET.SubElement(lanelet, 'adjacentRight', ref=str(idx + 2), drivingDir="same")
            
    if os.path.exists(initialization_path):
        init_tree = ET.parse(initialization_path)
        init_root = init_tree.getroot()
        
        vehicle_types = parse_vehicle_types(rou_file_path)
        veh_idx = 30
        for vehicle in init_root.findall('vehicle'):
            veh_idx += 1
            vehicle_id = vehicle.get('id')
            vehicle_type = vehicle.find('type').text
            length, width = vehicle_types.get(vehicle_type, ("5.0", "2.0"))
            obstacle = ET.SubElement(root, 'obstacle', id=str(veh_idx))
            ET.SubElement(obstacle, 'role').text = "dynamic"
            ET.SubElement(obstacle, 'type').text = "car"
            shape = ET.SubElement(obstacle, 'shape')
            rectangle = ET.SubElement(shape, 'rectangle')
            ET.SubElement(rectangle, 'length').text = length
            ET.SubElement(rectangle, 'width').text = width
            trajectory = ET.SubElement(obstacle, 'trajectory')
            state = ET.SubElement(trajectory, 'state')

            lane_id = vehicle.find('lane').text
            x, y = position_along_lane_to_xy(lane_id_to_points[lane_id], float(vehicle.find('pos').text))
            position = ET.SubElement(state, 'position')
            point = ET.SubElement(position, 'point')
            ET.SubElement(point, 'x').text = f'{x:.4f}'
            ET.SubElement(point, 'y').text = f'{y:.4f}'
            orientation = ET.SubElement(state, 'orientation')
            ET.SubElement(orientation, 'exact').text = vehicle.find('orientation').text
            time = ET.SubElement(state, 'time')
            ET.SubElement(time, 'exact').text = "0"
            velocity = ET.SubElement(state, 'velocity')
            ET.SubElement(velocity, 'exact').text = vehicle.find('speed').text
            acceleration = ET.SubElement(state, 'acceleration')
            ET.SubElement(acceleration, 'exact').text = vehicle.find('acc').text

    problem_tree = ET.parse(planning_problem_path)
    problem_root = problem_tree.getroot()
    planning_problems = problem_root.findall('planningProblem')
    for planning_problem in planning_problems:
        root.append(planning_problem)

    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    pretty_xml_str = "\n".join([line for line in pretty_xml_str.split("\n") if line.strip()])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)

def generate_cr_map(root_path, force_generate=False):
    '''generate commonroad map from SUMO map, initialization and planning problem'''

    sumo_map_path = os.path.join(root_path,'road.net.xml')
    commonroad_map_path = os.path.join(root_path, 'cr_map.xml')
    initialization_path = os.path.join(root_path, 'initialization.xml')
    planning_problem_path = os.path.join(root_path, 'planning_problem.xml')
    rou_file_path = os.path.join(root_path, 'vehicle.rou.xml')
    # road point interval
    interval = 10 
    
    #if not os.path.exists(initialization_path) or force_generate:
    #    generate_simple_initialization(initialization_path)
    if not os.path.exists(planning_problem_path) or force_generate:
        generate_planning_problem_xml(planning_problem_path)
    if not os.path.exists(commonroad_map_path) or force_generate:
        lanes = parse_sumo_map(sumo_map_path, interval)
        generate_commonroad_map(lanes, commonroad_map_path, initialization_path, planning_problem_path, rou_file_path)

if __name__ == '__main__':
    generate_cr_map('/home/linxuan/Formal/provable/code/benchmarks/CommonRoad/sumo_config/sumo_maps/3LaneHighway/', force_generate=True)