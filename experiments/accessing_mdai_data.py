import json
import yaml
import mdai  # Need to install mdai library via pip
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from PIL import ImageDraw
# from wand.image import Image
# from wand.drawing import Drawing
# from wand.color import Color
# from scipy.ndimage import binary_fill_holes
from datetime import datetime
from collections import Counter, deque, defaultdict

from experiments.utils import load_config


def interpret_free_fluid_label(label_name):
    """
    This function interprets the label name for the free fluid label and returns the corresponding label.
    :param label_name:
    :return: 1 (free fluid) or -1 (no free fluid)
    """
    if label_name == 'Mask-Morison FreeFluid':
        return 1
    elif label_name == 'Mask-MorisonPouch':
        return -1
    else:
        raise ValueError(f"Label {label_name} not recognized.")


def get_most_recent_json_file(directory):
    """
    This function gets the most recent json file in a directory.
    :param directory: str, directory
    :return: str, most recent json file
    """
    json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
    #
    most_recent_file = None
    most_recent_date = None
    #
    for file in json_files:
        date_string = file.split('_')[-1].split('.')[0]
        date = datetime.strptime(date_string, '%Y-%m-%d-%H%M%S')
        #
        if most_recent_date is None or date > most_recent_date:
            most_recent_file = file
            most_recent_date = date
    #
    return most_recent_file


def get_annotations_from_mdai(args):
    """
    This function gets the full set of raw annotations from the mdai project and returns the relevant annotations.
    :param args: Dictionary of arguments with the following keys:
        DOMAIN: str, mdai domain
        PROJECT_ID: str, mdai project id
        IMAGE_DIR: str, directory to save data
        ACCESS_TOKEN: str, mdai access token
        DATASET_NAMES: list of str, names of datasets to get annotations from
        LABEL_GROUP: str, name of label group to get annotations from
        LABELS: list of str, names of masks that we want
    :return: list of dictionaries, each dictionary is a mask annotation
    """
    mdai_client = mdai.Client(domain=args['DOMAIN'], access_token=args['ACCESS_TOKEN'])
    p = mdai_client.project(args['PROJECT_ID'], path=args['IMAGE_DIR'])
    #
    json_filename = get_most_recent_json_file(args['IMAGE_DIR'])
    print('#################################################')
    print(f"Using json file {json_filename}.")
    with open(os.path.join(args['IMAGE_DIR'], json_filename), 'r') as file:
        data = json.load(file)
    #
    # Get the label group
    label_group = None
    for lg in data['labelGroups']:
        if lg['name'] == args['LABEL_GROUP']:
            label_group = lg
            break
    if label_group is None:
        raise ValueError(f"Label group {args['LABEL_GROUP']} not found in data.")
    else:
        print(f"Label group {args['LABEL_GROUP']} found in data.")
    args['LABEL_DICT'] = {l['id']: l['name'] for l in label_group['labels']}
    #
    # Get the label ids from the label group
    label_ids = []
    for l in label_group['labels']:
        if l['name'] in args['LABELS']:
            label_ids.append(l['id'])
    if label_group is None:
        raise ValueError(f"Labels {args['LABELS']} not found in data.")
    elif len(label_ids) != len(args['LABELS']):
        print(f"Warning: not all specified labels found in data.")
    else:
        print(f"Labels {args['LABELS']} found in data.")
    #
    # Get the mask annotations
    mask_annotations = []
    for ds in data['datasets']:
        if ds['name'] in args['DATASET_NAMES']:
            ds_counter = 0
            for annotation in ds['annotations']:
                if annotation['labelId'] in label_ids:
                    ds_counter += 1
                    annotation['dataset_name'] = ds['name']
                    mask_annotations.append(annotation)
            print(f"{ds_counter} mask annotations found in dataset {ds['name']}.")
    if len(mask_annotations) == 0:
        raise ValueError(f"No mask annotations found in data.")
    else:
        print(f"{len(mask_annotations)} total mask annotations found in data.")
    #
    return mask_annotations


def convert_annotation_to_image_and_label(annotation, args):
    # Use the SOPInstanceUID and frameNumber to create a filename for the image.
    mask_image_filename = f"masks/{annotation['SOPInstanceUID']}_{annotation['frameNumber']}_Mask.jpg"
    raw_image_filename = f"{annotation['SOPInstanceUID']}_{annotation['frameNumber']}.jpg"
    positive_label = interpret_free_fluid_label(args['LABEL_DICT'][annotation['labelId']])
    creator_id = annotation['createdById']
    dataset_name = annotation['dataset_name']
    study_id = annotation['StudyInstanceUID']

    frame_width = annotation['width']
    frame_height = annotation['height']

    # Flatten the array of 2D arrays
    flattened_array = np.concatenate(annotation['data']['foreground'][0])

    # Calculate the length of the flattened array
    length_list = len(flattened_array) // 2

    # Reshape the flattened array into (length_list, 2)
    list_in = flattened_array.reshape((length_list, 2))

    image = PILImage.new('RGB', (frame_width, frame_height), color='black')
    draw = ImageDraw.Draw(image)
    draw.polygon(list_in.flatten().tolist(), fill='white', outline='white', width=2)
    image.save(os.path.join(args['IMAGE_DIR'], mask_image_filename))
    # image.save(f"/scratch/users/austin.zane/ucsf_fast/data/labeled_fast_morison/debugging/PIL_saved_mask.jpg")

    return raw_image_filename, positive_label, creator_id, dataset_name, study_id


def process_annotations(annotations, args):
    """
    This function processes the annotations and returns the images and labels.
    :param annotations: list of dictionaries, mask annotations
    :return: list of dictionaries, images and labels
    """
    # Create the mask directory if it doesn't exist
    os.makedirs(os.path.join(args['IMAGE_DIR'], "masks"), exist_ok=True)

    label_list = []
    for annotation in annotations:
        image_and_label = convert_annotation_to_image_and_label(annotation, args)
        if image_and_label is not None:
            label_list.append(image_and_label)
    #
    csv_filename = os.path.join(args['IMAGE_DIR'], "free_fluid_labels.csv")
    #
    new_label_rows = pd.DataFrame(label_list,
                                  columns=['filename', 'free_fluid_label', 'creator_id', 'dataset_name', 'study_id'])
    #
    if os.path.exists(csv_filename):
        # Load the existing data
        existing_label_rows = pd.read_csv(csv_filename)
        # Append the new data
        updated_rows = pd.concat([existing_label_rows, new_label_rows], ignore_index=True)
    else:
        updated_rows = new_label_rows
    #
    ########################
    # Modified code begins #
    ########################
    # print('check 0')
    duplicates = updated_rows.duplicated(subset=['filename'], keep=False)  ###### Delete later
    duplicated_rows = updated_rows[duplicates]  ###### Delete later
    # print('check 1')

    previous_nrows = updated_rows.shape[0]
    updated_rows = updated_rows.drop_duplicates(subset=['filename'], keep='last')
    # updated_rows = updated_rows.drop_duplicates()  ######## Delete later
    new_nrows = updated_rows.shape[0]
    print(f"Removed {previous_nrows - new_nrows} duplicate rows based solely on filename.")
    # print('check 2')
    # Create the compare_rows DataFrame
    compare_rows = pd.DataFrame(columns=['filename', 'free_fluid_label0', 'creator_id0', 'dataset_name0',
                                         'free_fluid_label1', 'creator_id1', 'dataset_name1'])

    # Iterate over each row in duplicated_rows
    for _, row in duplicated_rows.iterrows():
        filename = row['filename']

        # Find the corresponding row in updated_rows
        updated_row = updated_rows.loc[updated_rows['filename'] == filename].iloc[0]

        # Create a new row for compare_rows
        # Create a new row for compare_rows
        new_row = {
            'filename': [filename],
            'free_fluid_label0': [updated_row['free_fluid_label']],
            'creator_id0': [updated_row['creator_id']],
            'dataset_name0': [updated_row['dataset_name']],
            'free_fluid_label1': [row['free_fluid_label']],
            'creator_id1': [row['creator_id']],
            'dataset_name1': [row['dataset_name']]
        }

        # Append the new row to compare_rows
        compare_rows = pd.concat([compare_rows, pd.DataFrame(new_row)], ignore_index=True)
    # print('check3')
    #
    updated_rows.to_csv(csv_filename, index=False)  ######## Uncomment later

    ########################
    # Modified code ends   #
    ########################
    #
    # Counting the occurrences of postive and negative annotations
    val_counts = updated_rows['free_fluid_label'].value_counts()
    print(f'Number of positive images: {val_counts.get(1, 0)}')
    print(f'Number of negative images: {val_counts.get(-1, 0)}')
    #

    return updated_rows, duplicated_rows, compare_rows


def collect_and_rename_images(updated_rows, args):
    """
    This function collects the images downloaded from, renames them, and copies them to the new image directory.
    The images come from two directories with different naming conventions, so this function has two parts.

    :param updated_rows: DataFrame, images and labels
    :param args: Dictionary, arguments
    :return: None
    """
    ##########
    # PART I #
    ##########
    print('Processing RUQ raw images.')

    # This is the location of the directory downloaded from RAE at M:\TempAnnotations_Morison\DCMFRM\DCMFRM
    source_dir = args['RUQ_DIR']

    destination_dir = os.path.join(args['IMAGE_DIR'], 'raw_images')

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over the files in the source directory
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".jpg"):
            # Extract the SOPInstanceUID and frameNumber from the filename
            parts = file_name.split("_")
            sop_instance_uid = parts[0].replace(".dcm", "")
            frame_number = parts[1].replace(".jpg", "")
            frame_number = str(int(frame_number) - 1)

            # Create the new filename
            new_file_name = f"{sop_instance_uid}_{frame_number}.jpg"

            # Copy the file to the destination directory with the new filename
            new_file_path = os.path.join(destination_dir, new_file_name)

            if os.path.exists(new_file_path):
                continue

            source_path = os.path.join(source_dir, file_name)
            shutil.copy(source_path, new_file_path)

    ###########
    # PART II #
    ###########

    print('Processing P1315 raw images.')

    # This is the location of the directory downloaded from RAE at M:\Positive_13_15\P1315_Images
    source_dir = args['P1315_DIR']
    source_inst_rec = pd.read_csv(args['P1315_INSTANCE_RECORD'])

    source_inst_rec['Subdirectory'] = source_inst_rec['Filename'].apply(lambda x: x.split('/')[0])
    source_inst_rec['Prefix'] = source_inst_rec['Filename'].apply(lambda x: x.split('/')[1].split('.')[0])

    # Group by subdirectory
    subdir_groups = source_inst_rec.groupby('Subdirectory')

    for subdir, group in subdir_groups:
        subdir_path = os.path.join(source_dir, subdir)
        if not os.path.exists(subdir_path):
            print('WARNING: batch subdirectory in InstanceRecord.csv for P1315 not found.')
            continue  # Skip if subdir does not exist

        # List all files in the subdirectory
        existing_files = os.listdir(subdir_path)

        # Process each prefix in the group
        for index, row in group.iterrows():
            file_prefix = row['Prefix']
            sop_instance_uid = row['SOPInstanceUID']

            # Filter and process files matching the prefix
            matching_files = [f for f in existing_files if f.startswith(file_prefix) and f.endswith('.jpg')]
            for file in matching_files:
                frame_number = file.split('_')[-1].split('.')[0]
                frame_number = str(int(frame_number) - 1)

                # Create new file name using SOPInstanceUID and retain the suffix
                new_file_name = f"{sop_instance_uid}_{frame_number}.jpg"
                new_file_path = os.path.join(destination_dir, new_file_name)

                if os.path.exists(new_file_path):
                    continue

                # Copy and rename the file
                source_path = os.path.join(subdir_path, file)
                shutil.copy(source_path, new_file_path)


    ############
    # Part III #
    ############

    if not os.path.exists(os.path.join(args['IMAGE_DIR'], 'masks')):
        print('WARNING: no mask directory detected.')
        return None

    mask_file_names = os.listdir(os.path.join(args['IMAGE_DIR'], 'masks'))
    raw_image_file_names = [x.split('_Mask.jpg')[0] + '.jpg' for x in mask_file_names]

    num_no_match = 0
    file_names_no_match = []

    for raw_image_file_name in raw_image_file_names:
        if not os.path.exists(os.path.join(args['IMAGE_DIR'], 'raw_images', raw_image_file_name)):
            num_no_match += 1
            file_names_no_match.append(raw_image_file_name)

    print(f'Number of masks with no matching raw image: {num_no_match}')
    for f in file_names_no_match:
        print(f)

    # All of the masks with no matching raw number have 0 index. Could be a much larger problem.

    return None

#######################
# Debugging functions #
#######################

def investigate_indexing_mismatch():
    mask_file_names = os.listdir('/scratch/users/austin.zane/ucsf_fast/data/labeled_fast_morison/masks')
    mask_file_names = [x.split('_')[1] for x in mask_file_names]
    mask_counter = Counter(mask_file_names)
    print(f'Number of mask files with frame number zero: {mask_counter["0"]}')

    all_batch_file_names = []
    for batch_name in os.listdir('/scratch/users/austin.zane/ucsf_fast/data/P1315/P1315_Images'):
            if batch_name == '.DS_Store':
                continue
            batch_file_names = os.listdir(
                os.path.join('/scratch/users/austin.zane/ucsf_fast/data/P1315/P1315_Images', batch_name)
            )
            batch_file_names = [x.split('_')[1].split('.')[0] for x in batch_file_names]
            all_batch_file_names.extend(batch_file_names)

    batch_counter = Counter(all_batch_file_names)
    print(f'Number of P1315 files with frame number zero: {batch_counter["0"]}')

    all_ruq_file_names = os.listdir('/scratch/users/austin.zane/ucsf_fast/data/pilot_labeling/DCMFRM')
    all_ruq_file_names = [x.split('_')[1].split('.')[0] for x in all_ruq_file_names]
    ruq_counter = Counter(all_ruq_file_names)
    print(f'Number of RUQ files with frame number zero: {ruq_counter["0"]}')

    return None


def find_cluster_sizes(array):
    """
    Find all clusters of '1's in the array and record the frequency of each group size.
    If the mask outline is continuous, this function will return a single cluster size.

    :param array: 2D numpy array of 0s and 1s
    :return: Dictionary where keys are group sizes and values are frequencies
    """
    rows, cols = array.shape
    visited = np.zeros_like(array, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # N, S, W, E
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # NW, NE, SW, SE

    cluster_sizes = defaultdict(int)

    for i in range(rows):
        for j in range(cols):
            if array[i, j] == 1 and not visited[i, j]:
                # Start BFS from the unvisited '1' cell
                queue = deque([(i, j)])
                visited[i, j] = True
                cluster_size = 1

                while queue:
                    r, c = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and array[nr][nc] == 1:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cluster_size += 1

                # Record the frequency of the cluster size
                cluster_sizes[cluster_size] += 1

    print("Cluster Size Frequencies:")
    for size, freq in sorted(cluster_sizes.items()):
        print(f"Size {size}: {freq} {'cluster' if freq == 1 else 'clusters'}")
    print()
    return cluster_sizes


def print_dict(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def main():

    mdai_args = load_config()

    annotations = get_annotations_from_mdai(mdai_args)
    #
    updated_rows, duplicated_rows, compare_rows = process_annotations(annotations, mdai_args)
    #
    collect_and_rename_images(updated_rows, mdai_args)
    #
    # print(updated_rows)
    #
    return 0

if __name__ == "__main__":
    main()