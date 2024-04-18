# Need to install mdai library via pip

import mdai

DOMAIN = "ucsf.md.ai"
PROJECT_ID = "7YNd8LNb"
DATASET_IDS = [
  "D_pQ7xYz",
  "D_aQB9GJ",
  "D_rVgbwz"
]

# create and insert your access token here
ACCESS_TOKEN = ""

mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)

# Example project initialization - see documentation for details.
# Add dataset_id to scope project to a specific dataset.
# Change path to use a different directory.
p = mdai_client.project(PROJECT_ID, path="./data")


import json

with open("/accounts/campus/austin.zane/ucsf_fast/notebooks/data/mdai_ucsf_project_7YNd8LNb_annotations_2024-04-17-213920.json", 'r') as file:
    data = json.load(file)


label_info_data = data['labelGroups'][0]['labels'][0:2]
label_info_data


label_info_data = data['labelGroups'][10]['labels'][0:2]
label_of_interest = ['L_7vzYO7', 'L_1ma6G1']

full_study_data = data['datasets'][10]

annotation_data = full_study_data['annotations']

mask_present = [False] * len(annotation_data)
free_fluid_present = [False] * len(annotation_data)

for idx in range(len(annotation_data)):
    mask_present[idx] = annotation_data[idx]['labelId'] == label_of_interest[0]
    free_fluid_present[idx] = annotation_data[idx]['labelId'] == label_of_interest[1]
