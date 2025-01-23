# Visualization

Basic visualization functions are provided in the `render_video` function under the `commonroad_sumo_env.py`, and videos are automatically saved in `results/saved_data/videos` after testing. 

You can also run 
```bash
python NADE_result_analysis.py --root_folder <path-to-your-result-folder> --visualize_flag
```
to generate your own visualization from the saved data. Videos are saved in `root_folder/saved_data/videos`.

Some visualization examples:

Unsafe planner

![unsafe](figure/demo_unsafe_planner.gif)

Safe planner 

![safe](figure/demo_safe_planner.gif)

Safe driver 

![safedriver](figure/demo_safe_driver.gif)

<- Last Page: [Testing](testing.md)