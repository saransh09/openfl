from pathlib import Path
from openfl.federated import Plan
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--name', metavar='name', required=True, help='Enter the name of the collaborator')
args = parser.parse_args()

plan_path = './plan/plan.yaml'
data_path = './data.yaml'

plan = Plan.Parse(plan_config_path=Path(plan_path), data_config_path=Path(data_path))
plan.get_collaborator(args.name).run()