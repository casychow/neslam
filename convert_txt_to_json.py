import json
import os

"""
Purpose of this module is to convert ORB-SLAM output text file into a json file
that NerfStudio/NerfStudio's COLMAP can run.

Following directions from COLMAP's official documentation:
https://colmap.github.io/faq.html?highlight=two%20view%20tracks#reconstruct-sparse-dense-model-from-known-camera-poses
https://colmap.github.io/format.html#binary-file-format

Assuming we are using the same recorded data for COLMAP and ORB-SLAM (ie. same camera),
we want to create a dense model from known camera poses (taken from ORB-SLAM)

Requirements/Directions:
- Need to put the output transforms.json file in the same parent directory as "colmap"
and the "images (_2, _4, _8) folder" before you feed it into NerfStudio
- Be sure to save COLMAP transforms.json file somewhere else if you want it.
Otherwise this program will overwrite the file.
- Change the ORBSLAM_TXT_OUTPUT_FILENAME variable to the ORB-SLAM output file name.

Folder hierarchy:
-processed-data (custom folder name)
---colmap
---images
---images_2
---images_4
---images_8
---transforms.json
"""

# def create_transform_matrix()

ORBSLAM_TO_NERFSTUDIO = True
ORBSLAM_TXT_OUTPUT_FILENAME = 'f_dataset-MH01_monoi.txt'

if not ORBSLAM_TO_NERFSTUDIO:
    # Need to create cameras.txt, points3D.txt, and images.txt in a new folder for COLMAP
    # Create new directory and cameras.txt and 
    main_cwd = os.getcwd()
    folder = 'orbslam_scene_model'

    # If orbslam_scene_model folder already exists, change directory to that
    folder_path = os.path.join(main_cwd, folder)
    if os.path.exists(folder_path):
        os.chdir(folder_path)
    else:
        # Make directory
        os.mkdir(folder_path)

    # Edit cameras.txt file
    cparam_file = open('cameras.txt', 'w')
    camera_id = ''
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # 1         PINHOLE  640     480    382.613 (focal length) 320.183 (principal point x) 236.455 (principal point y)
    cparam_file.write("1 PINHOLE 640 480 382.613 320.183 236.455")
    cparam_file.close()

try:
    # [] - Automate adding info from json file
    data_dict = {
        "fl_x": 382.613,
        "fl_y": 382.613,
        "cx": 320.183,
        "cy": 236.455,
        "w": 640,
        "h": 480,
        "camera_model": "PINHOLE", #might be 'OPENCV' instead
        "k1": 0, #-0.04320926712496241,
        "k2": 0, #0.03283919508795189,
        "p1": 0, #-0.003930289929289693,
        "p2": 0, #0.001183353703859939,
        "frames": []
    }

    # https://colmap.github.io/format.html#binary-file-format
    # Edit images.txt file for COLMAP using data from ORB-SLAM
    # The format of each ORB-SLAM line is 'timestamp tx ty tz qx qy qz qw'
        # timestamp (float) gives the number of seconds since the Unix epoch.
        # tx ty tz (3 floats) give the position of the optical center of the color camera with respect to the world origin as defined by the motion capture system.
        # qx qy qz qw (4 floats) give the orientation of the optical center of the color camera in form of a unit quaternion with respect to the world origin as defined by the motion capture system.
    # Format for each images.txt line is 'IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                                        # POINTS2D[] as (X, Y, POINT3D_ID)'
    # images_file = open('images.txt', 'w')
    with open(ORBSLAM_TXT_OUTPUT_FILENAME) as txt_file:
        for ind, line in enumerate(txt_file):
            tx, ty, tz, qx, qy, qz, qw = line.split()[1:]
            tx = float(tx)
            ty = float(ty)
            tz = float(tz)
            qx = float(qx)
            qy = float(qy)
            qz = float(qz)
            qw = float(qw)
            # Only need to rewrite data in the correct format if we use COLMAP again
            # if not ORBSLAM_TO_NERFSTUDIO: colmap_data = [ind+1, qw, qx, qy, qz, tx, ty, tz, 1, 'PINHOLE']

            # Construct 'file_path' value
            digit = len(str(ind + 1))
            image_format_zeroes = 5 - digit # Count number of zeroes we need to specify frame number
            image_path = "images\\frame_{}.jpg".format(image_format_zeroes*'0'+str(ind+1))

            # Construct 'transform_matrix' value
            R_matrix = [
                [1-2*qz**2-2*qw**2, 2*qy*qz-2*qx*qw, 2*qy*qw+2*qx*qz, tx],
                [2*qy*qz+2*qx*qw, 1-2*qy**2-2*qw**2, 2*qz*qw-2*qx*qy, ty],
                [2*qy*qw-2*qx*qz, 2*qz*qw+2*qx*qy, 1-2*qy**2-2*qz**2, tz],
                [0.0, 0.0, 0.0, 1.0]
            ]

            # Construct frame_dict to append to 'frames' value
            frame_dict = {
                "file_path": image_path,
                "transform_matrix": R_matrix
            }
            data_dict['frames'].append(frame_dict)


    # function
    #tx-tz last col
    #qx-qw = q1-q4 for R matrix
    #last row of R matrix is homogeneous (0,0,0,1)


    with open('transforms.json', 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)

    # For COLMAP:
    # Read camera poses
    # Need to write camera calibration information first
except FileNotFoundError:
    print(f"File {ORBSLAM_TXT_OUTPUT_FILENAME} not found. Aborting")
except OSError:
    print(f"OS error occurred trying to open {ORBSLAM_TXT_OUTPUT_FILENAME}")
except Exception as err:
    print(f"Unexpected error opening {ORBSLAM_TXT_OUTPUT_FILENAME} is", repr(err))

"""
format of json file:
{
    "fl_x": 386.77627031025804,
    "fl_y": 387.0815437667829,
    "cx": 319.556497088614,
    "cy": 244.210178136688,
    "w": 640,
    "h": 480,
    "camera_model": "OPENCV",
    "k1": -0.04320926712496241,
    "k2": 0.03283919508795189,
    "p1": -0.003930289929289693,
    "p2": 0.001183353703859939,
    "frames": [
        {
            "file_path": "images\\frame_00944.jpg",
            "transform_matrix": [
                [
                    0.2231971415763487,
                    -0.9588444697581704,
                    0.1755001960292933,
                    -0.5456329668801857
                ],
                [
                    0.8525012253216967,
                    0.2793146619321971,
                    0.4418427100843786,
                    0.5479332166190242
                ],
                [
                    -0.4726782169903253,
                    0.05099610224198986,
                    0.8797583195054035,
                    6.0685035316099265
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
            #3x4
        }
    ]
}
"""