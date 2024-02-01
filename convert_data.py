"""
Convert the HOPE-Image dataset to the BOP format
"""

import glob
import os
import numpy as np
import json
import cv2
import open3d as o3d

old_format_path = '/home/gouda/segmentation/making_HOPE_BOP/HOPE-Image_Github'
bop_format_save_path = '/home/gouda/segmentation/making_HOPE_BOP/hope_video'
bop_models_path = '/home/gouda/segmentation/making_HOPE_BOP/hope_models/models'


#obj_ordered_list = [AlphabetSoup,BBQSauce,Butter,Cherries,ChocolatePudding,Cookies,Corn,CreamCheese,GranolaBars,GreenBeans,Ketchup,MacaroniAndCheese,Mayo,Milk,Mushrooms,Mustard,OrangeJuice,Parmesan,Peaches,PeasAndCarrots,Pineapple,Popcorn,Raisins,SaladDressing,Spaghetti,TomatoSauce,Tuna,Yogurt]
obj_ordered_list = ['AlphabetSoup', 'BBQSauce', 'Butter', 'Cherries', 'ChocolatePudding', 'Cookies', 'Corn',
                    'CreamCheese', 'GranolaBars', 'GreenBeans', 'Ketchup', 'MacaroniAndCheese', 'Mayo', 'Milk',
                    'Mushrooms', 'Mustard', 'OrangeJuice', 'Parmesan', 'Peaches', 'PeasAndCarrots', 'Pineapple',
                    'Popcorn', 'Raisins', 'SaladDressing', 'Spaghetti', 'TomatoSauce', 'Tuna', 'Yogurt']
obj_name_dict = {}
for i in range(len(obj_ordered_list)):
    obj_name_dict[obj_ordered_list[i]] = i+1

# all images are 640x480
width, height = 640, 480

# iterate over each of the 10 scenes
for scene_num in range(10):
    old_scene_path = os.path.join(old_format_path, 'hope_video_scene_' + f'{scene_num:04}')
    bop_scene_path = os.path.join(bop_format_save_path, f'{scene_num:06}')
    # make folders rgb, depth, mask, mask_vis
    os.makedirs(os.path.join(bop_scene_path, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(bop_scene_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(bop_scene_path, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(bop_scene_path, 'mask_vis'), exist_ok=True)
    # define json files data
    scene_camera = {}
    scene_gt = {}
    scene_gt_info = {}

    num_of_frames = glob.glob(os.path.join(old_scene_path, '*.jpg'))
    # iterate over each frame
    for frame_num in range(len(num_of_frames)):
        # copy rgb image to bop folder
        old_rgb_frame = os.path.join(old_scene_path, f'{frame_num:04}_rgb.jpg')
        bop_rgb_frame = os.path.join(bop_scene_path, 'rgb', f'{frame_num:06}.jpg')
        os.system(f'cp {old_rgb_frame} {bop_rgb_frame}')
        # read depth image, convert it from cm to m, then save it to bop folder
        old_depth_frame = os.path.join(old_scene_path, f'{frame_num:04}_depth.png')
        # read depth image
        frame_depth_image = cv2.imread(old_depth_frame, cv2.IMREAD_ANYDEPTH)  # in mm
        # save depth image
        bop_depth_frame = os.path.join(bop_scene_path, 'depth', f'{frame_num:06}.png')
        cv2.imwrite(bop_depth_frame, frame_depth_image)

        # read json file, note json has '\n' newline characters
        old_json_file = os.path.join(old_scene_path, f'{frame_num:04}.json')
        with open(old_json_file, 'r') as f:
            json_data = json.load(f)
        # read camera extrinsics
        camera_extrinsics = json_data["camera"]["extrinsics"]
        camera_extrinsics = np.array(camera_extrinsics).reshape(4,4)
        # read camera intrinsics
        camera_intrinsics = json_data["camera"]["intrinsics"]
        camera_intrinsics = np.array(camera_intrinsics).reshape(3,3)
        # make scene_camera (extrinsics, intrinsics)
        frame_cam = {}
        frame_cam["cam_k"] = camera_intrinsics.flatten().tolist()
        frame_cam["depth_scale"] = 1.0
        translation = camera_extrinsics[:3,3]
        rotation = camera_extrinsics[:3,:3]
        frame_cam["cam_t_w2c"] = translation.tolist()
        frame_cam["cam_R_w2c"] = rotation.flatten().tolist()
        scene_camera[frame_num] = frame_cam
        # make scene_gt
        # read object list
        object_list = json_data["objects"]
        # rewrite object list with the corresponding object id from the ordered list
        frame_gt = []
        for obj in object_list:
            obj_data = {}
            obj_data["obj_id"] = obj_name_dict[obj["class"]]
            pose = obj["pose"]  # in cm
            pose = np.array(pose).reshape(4,4)
            translation = pose[:3,3] * 10  # convert from cm to mm
            rotation = pose[:3,:3]
            # change rotation to xyz from zyx
            rotation_flip = np.array([[0, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 0]])
            rotation = rotation @ rotation_flip
            obj_data["cam_t_m2c"] = translation.tolist()
            obj_data["cam_R_m2c"] = rotation.tolist()
            frame_gt.append(obj_data)
        scene_gt[frame_num] = frame_gt

        # ==============================================================================================================
        # # Debugging: generate point cloud of the frame using Open3D
        # # read rgb image
        rgb_image_o3d = o3d.io.read_image(bop_rgb_frame)
        depth_image_o3d = o3d.io.read_image(bop_depth_frame)
        rgbd_image_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, depth_image_o3d, depth_scale=1.0, depth_trunc=10000.0, convert_rgb_to_intensity=False)
        # convert camera intrinsics to open3d format
        camera_intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2])
        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_o3d, camera_intrinsics_o3d)
        # transform point cloud to the world coordinate frame
        frame_pcd.transform(np.linalg.inv(camera_extrinsics))
        # make origin coordinate frame at the world frame
        origin_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        # make another coordinate frame at the camera frame
        camera_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
        camera_mesh_frame.transform(np.linalg.inv(camera_extrinsics))
        # visualize point cloud
        o3d.visualization.draw_geometries([frame_pcd, origin_mesh_frame, camera_mesh_frame])
        # loop through each object in frame_gt, load its model, transform the object then visualize all objects
        objects_visualize = []
        for obj in frame_gt:
            obj_id = obj["obj_id"]
            # load object point cloud
            obj_pcd = o3d.io.read_triangle_mesh(os.path.join(bop_models_path, f'obj_{obj_id:06}.ply'))
            # transform object point cloud
            H_m2c = np.eye(4)
            H_m2c[:3,3] = np.array(obj["cam_t_m2c"])
            H_m2c[:3,:3] = np.array(obj["cam_R_m2c"])
            # transform object to the world coordinate frame
            H_w2c = camera_extrinsics
            H_c2w = np.linalg.inv(H_w2c)
            H_m2w = H_c2w @ H_m2c

            obj_pcd.transform(H_m2w)
            objects_visualize.append(obj_pcd)
        o3d.visualization.draw_geometries(objects_visualize + [frame_pcd, origin_mesh_frame, camera_mesh_frame])
        # ==============================================================================================================

        # make open3d Scene, to help generating depth images and masks
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, left=50, top=50, visible=True)
        ctr = vis.get_view_control()

        # add any of the objects just for testing
        #vis.add_geometry(frame_pcd, reset_bounding_box=True)
        #vis.poll_events()
        #vis.update_renderer()
        #vis.remove_geometry(frame_pcd)

        # make visualizer look at z where up is -y
        ctr.set_lookat([0, 0, 1])
        ctr.set_up([0, -1, 0])

        pinhole_camera_parameters = o3d.camera.PinholeCameraParameters()
        camera_intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, camera_intrinsics[0,0], camera_intrinsics[1,1], camera_intrinsics[0,2], camera_intrinsics[1,2])
        pinhole_camera_parameters.intrinsic = camera_intrinsics_o3d
        pinhole_camera_parameters.extrinsic = np.linalg.inv(camera_extrinsics)

        # set point size in the visualizer to 1
        opt = vis.get_render_option()
        opt.point_size = 3
        opt.show_coordinate_frame = True

        # generate mask for each object in frame_gt
        frame_masks = []
        for anno_id in range(len(frame_gt)):
            # load object mesh and transform it to the world coordinate frame
            obj_id = frame_gt[anno_id]["obj_id"]
            obj_mesh = o3d.io.read_triangle_mesh(os.path.join(bop_models_path, f'obj_{obj_id:06}.ply'))
            H_m2c = np.eye(4)
            H_m2c[:3,3] = np.array(frame_gt[anno_id]["cam_t_m2c"])
            H_m2c[:3,:3] = np.array(frame_gt[anno_id]["cam_R_m2c"])

            # transform object to the world coordinate frame
            H_w2c = camera_extrinsics
            H_c2w = np.linalg.inv(H_w2c)
            H_m2w = H_c2w @ H_m2c

            obj_mesh.transform(H_m2w)
            vis.add_geometry(obj_mesh, reset_bounding_box=True)
            # render the scene
            vis.poll_events()
            vis.update_renderer()

            #o3d.visualization.draw_geometries([obj_mesh, frame_pcd ,origin_mesh_frame, camera_mesh_frame])

            # this should be outside the loop
            ctr.convert_from_pinhole_camera_parameters(pinhole_camera_parameters, allow_arbitrary=True)
            vis.poll_events()
            vis.update_renderer()


            # capture depth image
            fake_depth_image_o3d = vis.capture_depth_float_buffer(do_render=False)
            # convert depth image to numpy array
            fake_depth_image = np.asarray(fake_depth_image_o3d)
            # generate mask
            mask = np.zeros((height, width))
            mask[fake_depth_image > 0] = 255
            # save mask
            mask_file = os.path.join(bop_scene_path, 'mask', f'{frame_num:06}_{anno_id:06}.png')
            cv2.imwrite(mask_file, mask)
            # append mask to frame_masks
            frame_masks.append(mask)
            # remove object from scene
            vis.clear_geometries()

        # generate mask_vis for each object in frame_gt
        frame_masks_vis = []
        # add all objects to scene to generate a fake depth image
        for obj in objects_visualize:
            vis.add_geometry(obj, reset_bounding_box=False)
        # capture the fake depth image
        fake_depth_image_o3d = vis.capture_depth_float_buffer(do_render=True)
        # convert depth image to numpy array
        fake_depth_image = np.asarray(fake_depth_image_o3d)
        # loop through each object in frame_gt
        for anno_id, mask in enumerate(frame_masks):
            # generate mask_vis
            mask_vis = np.zeros((height, width))
            # mask_vis = intersection between mask and fake_depth_image
            mask_vis[np.logical_and(mask, fake_depth_image > 0)] = 255
            # save mask_vis
            mask_vis_file = os.path.join(bop_scene_path, 'mask_vis', f'{frame_num:06}_{anno_id:06}.png')
            cv2.imwrite(mask_vis_file, mask_vis)
            # append mask_vis to frame_masks_vis

        # make scene_gt_info
        # all bboxes are (x, y, width, height)
        frame_gt_info = []
        for anno_id in range(len(frame_masks)):
            frame_obj_info = {}
            # get bounding box of mask
            bbox = list(cv2.boundingRect(frame_masks[anno_id]))
            frame_obj_info["bbox_obj"] = bbox
            # get bounding box of mask_vis
            bbox_vis = list(cv2.boundingRect(frame_masks_vis[anno_id]))
            frame_obj_info["bbox_visib"] = bbox_vis
            # get number of pixels in mask
            frame_obj_info["px_count_all"] = np.sum(frame_masks[anno_id] > 0)
            # get numer of pixels in mask_vis
            frame_obj_info["px_count_visib"] = np.sum(frame_masks_vis[anno_id] > 0)
            # get fraction of pixels in mask_vis
            frame_obj_info["visib_fract"] = frame_obj_info["px_count_visib"] / frame_obj_info["px_count_all"]
            # get number of non-zero pixels in mask_vis over imposed on frame original depth image
            frame_obj_info["px_count_valid"] = np.sum(np.logical_and(frame_masks_vis[anno_id], frame_depth_image > 0))

    # save scene_camera
    scene_camera_file = os.path.join(bop_scene_path, 'scene_camera.json')
    with open(scene_camera_file, 'w') as f:
        json.dump(scene_camera, f)
    # save scene_gt
    scene_gt_file = os.path.join(bop_scene_path, 'scene_gt.json')
    with open(scene_gt_file, 'w') as f:
        json.dump(scene_gt, f)
    # save scene_gt_info
    scene_gt_info_file = os.path.join(bop_scene_path, 'scene_gt_info.json')
    with open(scene_gt_info_file, 'w') as f:
        json.dump(scene_gt_info, f)
