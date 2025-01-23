import argparse
from analysis_tools.analysis import analysis
from analysis_tools.clean_up_results_folder import clean
from analysis_tools.visualization import visualize

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--root_folder", type=str, default="")
parser.add_argument("--analysis_flag", type=bool, default=True)
parser.add_argument("--clean_flag", action="store_true")
parser.add_argument("--visualize_flag", action="store_true")
parser.add_argument("--all_flag", action="store_true")
args = parser.parse_args()
root_folder = args.root_folder

if args.all_flag:
    args.analysis_flag = True
    args.clean_flag = True
    args.visualize_flag = True

if args.analysis_flag:
    analysis(root_folder)
if args.clean_flag:
    clean(root_folder)
if args.visualize_flag:
    visualize(root_folder)