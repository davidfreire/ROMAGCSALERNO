import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import cv2

class XML_data():

    def __init__(self, filename):
        self.filename = filename

        if (os.path.isfile(filename)): # si existe, se abre y se coge el root
            self.doc = ET.parse(filename)
            self.root = self.doc.getroot()
        else: # se debe crear
            self.doc = None
            self.root = None

        # To draw instances annotations
        self.keypoint_connection_rules = [
        # face
        ("left_ear", "left_eye", (102, 204, 255)),
        ("right_ear", "right_eye", (51, 153, 255)),
        ("left_eye", "nose", (102, 0, 204)),
        ("nose", "right_eye", (51, 102, 255)),
        # upper-body
        ("left_shoulder", "right_shoulder", (255, 128, 0)),
        ("left_shoulder", "left_elbow", (153, 255, 204)),
        ("right_shoulder", "right_elbow", (128, 229, 255)),
        ("left_elbow", "left_wrist", (153, 255, 153)),
        ("right_elbow", "right_wrist", (102, 255, 224)),
        # lower-body
        ("left_hip", "right_hip", (255, 102, 0)),
        ("left_hip", "left_knee", (255, 255, 77)),
        ("right_hip", "right_knee", (153, 255, 204)),
        ("left_knee", "left_ankle", (191, 255, 128)),
        ("right_knee", "right_ankle", (255, 195, 77)),
        ]

        self.keypoint_names = (
            "nose",
            "left_eye", "right_eye",
            "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        )

        self.keypoint_threshold = 3 # 3%

    
# --------------------------------------TO SAVE GENERATED INSTANCES INTO XML--------------------------
    def add_vid_instance(self, predictions, id_frame):
        if self.root == None:
            self.root = ET.Element("Video")
            ET.SubElement(self.root, "ImgSize").text = str(predictions["instances"].to(torch.device('cpu'))[0].image_size)
        
        
        frame_data = ET.SubElement(self.root, "Frame")
        frame_data.text = str(id_frame)

        instance_group = ET.SubElement(frame_data, "Instances")

        if "instances" in predictions:
            #if (len(predictions["instances"]) > 0):
            #    ET.SubElement(self.root, "ImgSize").text = str(predictions["instances"].to(torch.device('cpu'))[0].image_size)

            for i in range(0, len(predictions["instances"])):
                instance_xml = ET.SubElement(instance_group, "Instance")
                instances = predictions["instances"].to(torch.device('cpu'))[i] # acceso individual
                if (instances.has("pred_classes")):
                    #print('clases', instances.pred_classes.numpy())
                    #print('scores', instances.scores.numpy())
                    ET.SubElement(instance_xml, "Class").text = str(instances.pred_classes.numpy())
                    ET.SubElement(instance_xml, "Score").text = str(instances.scores.numpy())
                if (instances.has("pred_boxes")):
                    bxs = instances.pred_boxes.tensor.numpy()
                    ET.SubElement(instance_xml, "Boxes").text = str( instances.pred_boxes.tensor.numpy()[0])
                    #print('boxes: \n', bxs[0])
                    #print(len(bxs[0]))
                if (instances.has("pred_keypoints")):
                    #print('keypoints: \n', instances.pred_keypoints.numpy()[0])
                    ET.SubElement(instance_xml, "Keypoints").text = str( instances.pred_keypoints.numpy()[0])
                    #print(len(instances.pred_keypoints.numpy()[0]))
                ###if (instances.has("pred_keypoint_heatmaps")):
                    #print('pred_keypoint_heatmaps: \n', instances.pred_keypoint_heatmaps.numpy()[0])
                    ###ET.SubElement(instance_xml, "Keypoints_heatmaps").text = str( instances.pred_keypoint_heatmaps.numpy()[0])
                    #print(len(instances.pred_keypoint_heatmaps.numpy()[0]))
    
    def add_img_instance(self, predictions):
        if self.root == None:
            self.root = ET.Element("Image")
            ET.SubElement(self.root, "ImgSize").text = str(predictions["instances"].to(torch.device('cpu'))[0].image_size)
        
        instance_group = ET.SubElement(self.root, "Instances")

        if "instances" in predictions:
            #if (len(predictions["instances"]) > 0):
            #    ET.SubElement(self.root, "ImgSize").text = str(predictions["instances"].to(torch.device('cpu'))[0].image_size)

            for i in range(0, len(predictions["instances"])):
                instance_xml = ET.SubElement(instance_group, "Instance")
                instances = predictions["instances"].to(torch.device('cpu'))[i] # acceso individual
                if (instances.has("pred_classes")):
                    #print('clases', instances.pred_classes.numpy())
                    #print('scores', instances.scores.numpy())
                    ET.SubElement(instance_xml, "Class").text = str(instances.pred_classes.numpy())
                    ET.SubElement(instance_xml, "Score").text = str(instances.scores.numpy())
                if (instances.has("pred_boxes")):
                    bxs = instances.pred_boxes.tensor.numpy()
                    ET.SubElement(instance_xml, "Boxes").text = str( instances.pred_boxes.tensor.numpy()[0])
                    #print('boxes: \n', bxs[0])
                    #print(len(bxs[0]))
                if (instances.has("pred_keypoints")):
                    #print('keypoints: \n', instances.pred_keypoints.numpy()[0])
                    ET.SubElement(instance_xml, "Keypoints").text = str( instances.pred_keypoints.numpy()[0])
                    #print(len(instances.pred_keypoints.numpy()[0]))
                ###if (instances.has("pred_keypoint_heatmaps")):
                    #print('pred_keypoint_heatmaps: \n', instances.pred_keypoint_heatmaps)
                    ####ET.SubElement(instance_xml, "Keypoints_heatmaps").text = str( instances.pred_keypoint_heatmaps.numpy()[0])
                    #print(len(instances.pred_keypoint_heatmaps.numpy()[0]))

    def save_xml(self):
        tree = ET.ElementTree(self.root)
        with open(self.filename, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>');
            tree.write(f, xml_declaration=False, encoding='utf-8')


# --------------------------------------TO READ GENERATED XML INSTANCES --------------------------
    def read_video_xml(self):
        assert(self.root != None)
        out_dict = []
        height = int(self.root.find('ImgSize').text[1:-1].split(',')[0].strip())
        width = int(self.root.find('ImgSize').text[1:-1].split(',')[1].strip())
        for idx, frame in enumerate(self.root.findall('.//Frame')): 
            frame_dict = {}
            frame_dict['ID'] = int(frame.text)       
            frame_dict['Instances'] = []
            for instance in frame.findall('.//Instances/Instance'):
                instance_dict = {}
                instance_dict['Class'] = np.fromstring(instance.find('Class').text[1:-1], sep=' ').astype(int)
                instance_dict['Score'] = np.fromstring(instance.find('Score').text[1:-1], sep=' ').astype(float)
                instance_dict['Boxes'] = np.fromstring(instance.find('Boxes').text[1:-1], sep=' ').astype(float)
                keypoints = [np.fromstring(kp.strip()[1:-1], sep=' ').astype(float) for kp in instance.find('Keypoints').text[1:-1].split('\n')]
                instance_dict['Keypoints'] = np.array(keypoints)
                frame_dict['Instances'].append(instance_dict)
            out_dict.append(frame_dict)
        return height, width, out_dict

    def read_image_xml(self):
        assert(self.root != None)
        out_dict = []
        height = int(self.root.find('ImgSize').text[1:-1].split(',')[0].strip())
        width = int(self.root.find('ImgSize').text[1:-1].split(',')[1].strip())

        frame_dict = {}
        frame_dict['ID'] = 0     
        frame_dict['Instances'] = []

        for instance in self.root.findall('.//Instances/Instance'):
            instance_dict = {}
            instance_dict['Class'] = np.fromstring(instance.find('Class').text[1:-1], sep=' ').astype(int)
            instance_dict['Score'] = np.fromstring(instance.find('Score').text[1:-1], sep=' ').astype(float)
            instance_dict['Boxes'] = np.fromstring(instance.find('Boxes').text[1:-1], sep=' ').astype(float)
            keypoints = [np.fromstring(kp.strip()[1:-1], sep=' ').astype(float) for kp in instance.find('Keypoints').text[1:-1].split('\n')]
            instance_dict['Keypoints'] = np.array(keypoints)
            frame_dict['Instances'].append(instance_dict)
        out_dict.append(frame_dict)
        return height, width, out_dict


# --------------------------------------TO GENERATE SOME INTEL (PRINT POINTS, EXTRACT PATCHES) --------------------------
    def draw_and_connect_keypoints(self, img, keypoints):
        """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.
        Args:
            keypoints: an array of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).
        Returns:
            res_img: image object with visualizations.
        """
        visible = {}
        res_img = img # img.copy()
        
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x = keypoint[0].astype(int)
            y = keypoint[1].astype(int)
            prob = (keypoint[2]*100).astype(int) 
    
            if prob > self.keypoint_threshold:
                res_img = cv2.circle(res_img, (x, y), 5, (255, 0, 0), -1)
                if self.keypoint_names:
                    keypoint_name = self.keypoint_names[idx]
                    visible[keypoint_name] = (x, y)

        # print(visible)
        for kp0, kp1, color in self.keypoint_connection_rules:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                #color = (0,0,0)
                thickness = 3
                res_img = cv2.line(res_img, (x0, y0), (x1, y1), color, thickness) 

        # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
        # Note that this strategy is specific to person keypoints.
        # For other keypoints, it should just do nothing
        try:
            ls_x, ls_y = visible["left_shoulder"]
            rs_x, rs_y = visible["right_shoulder"]
            mid_shoulder_x, mid_shoulder_y = int((ls_x + rs_x) / 2), int((ls_y + rs_y) / 2)
        except KeyError:
            pass
        else:
            # draw line from nose to mid-shoulder
            nose_x, nose_y = visible.get("nose", (None, None))
            if nose_x is not None:
                res_img = cv2.line(res_img, (nose_x, nose_y), (mid_shoulder_x, mid_shoulder_y), (255, 255, 255), thickness) 

            try:
                # draw line from mid-shoulder to mid-hip
                lh_x, lh_y = visible["left_hip"]
                rh_x, rh_y = visible["right_hip"]
            except KeyError:
                pass
            else:
                mid_hip_x, mid_hip_y = int((lh_x + rh_x) / 2), int((lh_y + rh_y) / 2)
                res_img = cv2.line(res_img, (mid_hip_x, mid_hip_y), (mid_shoulder_x, mid_shoulder_y), (255, 0, 0), thickness) 

        return res_img


    def draw_instances (self, img, detections):
        """
        For each image and a detection object it prints all boxes and keypoints connections.
        Args:
            img:  The original RGB image.
            detections: Instances detected as a dict from the xml file.

        Returns:
            label_img: image object with visualizations.
            instances_patches: images with each detections. 

        """  
        label_img = img.copy()

        instances_patches = {}
        instances_patches['images'] = []
        instances_patches['relative_position'] = [] # box centroid coordinates in src image
        instances_patches['keypoints'] = []

        for instance in detections['Instances']:

            box = instance['Boxes'].astype(int)
            label_img = cv2.rectangle(label_img, (box[0], box[1]), (box[2], box[3]), (255,0,255), 3)
            img_patch = img[box[1]:box[1]+(box[3]-box[1]),box[0]:box[0]+(box[2]-box[0])]

            instances_patches['relative_position'].append(np.array([ (box[2]-box[0])/2 + box[0], (box[3]-box[1])/2 + box[1]]))


            instances_patches['images'].append(img_patch)

            inst_kpt = []
            for keypoint in instance['Keypoints']:
                x = keypoint[0] - box[0]
                y = keypoint[1] - box[1]
                prob = keypoint[2]
                inst_kpt.append(np.array([x, y, prob]))

            instances_patches['keypoints'].append(inst_kpt)

            label_img = self.draw_and_connect_keypoints(label_img, instance['Keypoints'])
            
        return label_img, instances_patches

    
    def draw_image_instances(self, img_filename):
        """
        For an image and a xml filename it prints all boxes and keypoints connections but also returns the detections.
        Args:
            img_filename:  The original RGB image filename.
            xml_filename: xml file with annotations.

        Returns:
            label_img: image object with visualizations.
            instances_patches: images with detections.
        """  
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, detections = self.read_image_xml()

        # Check if it is the same image data extraction was computed in
        assert(img.shape[0] == h)
        assert(img.shape[1] == w)

        patches = {}

        patches['img_shape'] = np.array([h, w])
        label_img, instances_patches = self.draw_instances(img, detections[0])

        patches['instances'] = instances_patches

        return label_img, patches



    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def draw_video_instances(self, vid_filename, vid_out_filename):
        """
        For a video and a xml filename it prints all boxes and keypoints connections but also returns the detections.
        Args:
            vid_filename:  the original RGB video filename.
            xml_filename: xml file with annotations.
            vid_out_filename: filename for the output video.

        Returns:
            video_patches: frames with detections..
        """  

        #cap = cv2.VideoCapture()

        video = cv2.VideoCapture(vid_filename)
        h, w, detections = self.read_video_xml()

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check if it is the same image data extraction was computed in
        assert(height == h)
        assert(width == w)

        vid_patches = []
        

        output_file = cv2.VideoWriter(
                filename=vid_out_filename,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )

        frame_gen = self._frame_from_video(video)
        for id_frame, frame in enumerate(frame_gen):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            label_frame, instances_patches_frame = self.draw_instances(frame, detections[id_frame])
            label_frame = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
            output_file.write(label_frame)


            frame_patches = {}
            frame_patches['Frame_ID'] = id_frame
            frame_patches['instances'] = instances_patches_frame
            vid_patches.append(frame_patches)
        
        output_file.release()
        video.release()

        return vid_patches