import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt

class XML_data_deepsort():

    def __init__(self, filename):
        self.filename = filename

        if (os.path.isfile(filename)): # si existe, se abre y se coge el root
            self.doc = ET.parse(filename)
            self.root = self.doc.getroot()
        else: # se debe crear
            self.doc = None
            self.root = None

    
# --------------------------------------TO SAVE GENERATED INSTANCES INTO XML--------------------------
    def add_vid_instance(self, predictions, id_frame, sz_frame):
        if self.root == None:
            self.root = ET.Element("Video")
            ET.SubElement(self.root, "ImgSize").text = str(sz_frame)
        
        
        frame_data = ET.SubElement(self.root, "Frame")
        frame_data.text = str(id_frame)

        if (len(predictions)>0):
            instance_group = ET.SubElement(frame_data, "Detections")
            for instance in predictions:
                instance_xml = ET.SubElement(instance_group, "Instance")
                ET.SubElement(instance_xml, "Class").text = str(instance['class'])
                ET.SubElement(instance_xml, "ID").text = str(instance['id'])
                ET.SubElement(instance_xml, "BBox").text = str(instance['bbox'])
        
    
    def save_xml(self):
        tree = ET.ElementTree(self.root)
        with open(self.filename, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8" standalone="no"?>');
            tree.write(f, xml_declaration=False, encoding='utf-8')


    def read_video_xml(self):
        assert(self.root != None)
        out_dict = []
        width = int(self.root.find('ImgSize').text[1:-1].split(',')[0].strip())
        height = int(self.root.find('ImgSize').text[1:-1].split(',')[1].strip())
        for idx, frame in enumerate(self.root.findall('.//Frame')): 
            frame_dict = {}
            frame_dict['Frame_ID'] = int(frame.text)       
            frame_dict['Instances'] = []
            for instance in frame.findall('.//Detections/Instance'):
                instance_dict = {}
                instance_dict['Class'] = instance.find('Class').text
                instance_dict['Person_ID'] = int(instance.find('ID').text)
                instance_dict['BBox'] = np.fromstring(instance.find('BBox').text[1:-1], sep=',').astype(int)
                frame_dict['Instances'].append(instance_dict)
            out_dict.append(frame_dict)
        return height, width, out_dict

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

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        for instance in detections['Instances']:

            color = colors[int(instance['Person_ID']) % len(colors)]
            color = [i * 255 for i in color]

            box = instance['BBox']
            label_img = cv2.rectangle(label_img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.rectangle(label_img, (int(box[0]), int(box[1]-30)), (int(box[0])+(len(instance['Class'])+len(str(instance['Person_ID'])))*17, int(box[1])), color, -1)
            cv2.putText(label_img, instance['Class'] + "-" + str(instance['Person_ID']),(int(box[0]), int(box[1]-10)),0, 0.75, (255,255,255),2)

            img_patch = img[box[1]:box[1]+(box[3]-box[1]),box[0]:box[0]+(box[2]-box[0])]

            instances_patches['relative_position'].append(np.array([ (box[2]-box[0])/2 + box[0], (box[3]-box[1])/2 + box[1]]))

            instances_patches['images'].append(img_patch)

            
        return label_img, instances_patches


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