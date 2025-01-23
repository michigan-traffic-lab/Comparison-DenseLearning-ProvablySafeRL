import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_reach(reach, obstacles):
    '''Draw the occupancy of obstacles'''
    for o in obstacles:
        try:
            o.set.plot('r')
        except:
            pass
        if o.set.c.shape[0] > 1:
            pgon = o.set.polygon()
        for i in range(0, len(reach.occ)):
            if (o.time is None) or (o.time.intersects(reach.time[i])):
                try:
                    reach.occ[i].zonotope().plot('b')
                except:
                    pass

def draw_vehicle(ax, x, y, orientation, color, length=5, width=1.8):
    '''Draw the vehicle with given position, orientation, color, length and width'''
    vehicle = patches.Rectangle((-length/2, -width/2), length, width, linewidth=1, edgecolor=color, facecolor='none')
    t = patches.transforms.Affine2D().rotate_around(0, 0, orientation).translate(x, y) + ax.transData
    vehicle.set_transform(t)
    ax.add_patch(vehicle)

def draw(path, observation, scenario, reach, obstacles, info):
    '''Draw the scenario's reachable set'''
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    plt.close("all")
    ax = plt.subplot(2,1,1)
    max_x = observation['xPosition']
    min_x = observation['xPosition']
    draw_vehicle(ax, observation['xPosition'], observation['yPosition'], observation['orientation'], 'b')
    info_text = f"Ego state: x={observation['xPosition']:.2f}, y={observation['yPosition']:.2f}, velocity={observation['velocity']:.2f}, orientation={observation['orientation']:.2f} \n info: {info}"
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    for o in scenario.obstacles:
        pos = o.initial_state.position
        orientation = o.initial_state.orientation
        draw_vehicle(ax, pos[0], pos[1], orientation, 'r')
        if pos[0] > max_x:
            max_x = pos[0]
        if pos[0] < min_x:
            min_x = pos[0]
    x_range = max_x - min_x + 50
    plt.ylim(observation['yPosition']-x_range/6,observation['yPosition']+x_range/6)
    plt.xlim(min_x-10, max_x+40)
    plt.subplot(2,1,2)
    draw_reach(reach, obstacles)
    plt.ylim(-x_range/6,x_range/6)
    plt.xlim(min_x-observation['xPosition']-10,max_x-observation['xPosition']+40)
    plt.savefig(path,dpi=600)
    plt.close("all")