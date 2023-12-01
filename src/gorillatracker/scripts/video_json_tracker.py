import json
import cv2
import os
import subprocess


class GorillaVideoTracker:
    def __init__(self, path, outpath = "", videopath = "", allowed_overlap = 0.25):
        """ 
        initialize tracking of Individuals on videos with bounding boxes in json
        parameter: 
            path: path to json file(s)
            outpath: if not set output will be saved in path directory
            videopath: path to directory where videos are stored; if not set it will try to get videos from path
            allowed_overlap: % of the area of bboxes which is allowed to overlap between two individuals, default is 0.25
        """
        self.path = path
        self.directorypath = ""
        self.outpath = ""
        self.videopath = ""
        self.__mode = ""
        self.allowed_overlap = allowed_overlap
        
        if os.path.isdir(path):
            self.directorypath = path
            self.__mode = "directory"
        else:
            self.directorypath = os.path.dirname(path)
            self.__mode = "file"
            
        self.outpath = self.directorypath if outpath == "" else outpath   
        self.videopath = self.directorypath if videopath == "" else videopath    
              
    def track(self, log = True):
        """
        track individuals in path
        parameter:
            log: boolean; if progress should be logged to the terminal, default is True
        """
        if(self.__mode == "file"):
            files = [self.path]
        else:
            files = os.listdir(self.directorypath)
        file_count = len(files)  
        for idx, file in enumerate(files):
            if(idx % 5 == 0 and log is True):
                print(f"tracking...{idx}/{file_count}", end="\r")
            if os.path.splitext(file)[1].lower() != ".json":
                continue
            data = self.__readFromJson(os.path.join(self.directorypath, file))
            data, id_count = self.__trackIDs(data)
            data = self.__labelFaces(data)
            negatives = self.__getNegatives(data, id_count)
            self.__writeToJson(os.path.join(self.outpath, os.path.splitext(file)[0] + "_tracked.json"), data, negatives)
        if log is True: 
            print(f"{file_count} files successfully tracked")
              
    def saveVideos(self, max_video = 0, log = True):
        """
        save videos with bounding boxes for all tracked files
        parameter:
            log: boolean; if progress should be logged to the terminal, default is True            
            max_video: int, how many videos should be saved at maximum, 0 means no maximum, default is 0
            compress: boolean; if videofile should be compressed, default is True
        """
        tracked_files = [file for file in os.listdir(self.outpath) if file.endswith("_tracked.json")]
        file_count = len(tracked_files)
        if max_video == 0:
            max_video_idx = file_count + 1
        else:
            max_video_idx = max_video - 1
            file_count = max_video
            
        for idx, file in enumerate(tracked_files):
            if idx >= max_video_idx:
                break
            video_name = os.path.basename(file)[:-13]
            if log is True:
                print( " " * 80, end= "\r")
                print(f"saving video {idx + 1}/{file_count}: {video_name}.mp4", end = "\r")
            self.saveVideo(video_name = video_name, log = False)        
        if log is True:
            print( " " * 80, end= "\r")
            print(f"{file_count} videos successfully saved to {self.videopath}")
            
    def saveVideo(self, video_name = "", video_path = "", compress = True, log = True):
        """
        save video with bounding boxes
        parameter:
            video_name: name of video without path and extension e.g. for /path/to/example.mp4 just example
            video_path: path to videofile e.g. /path/to/example.mp4
            log: boolean; if progress should be logged to the terminal, default is True
            compress: boolean; if videofile should be compressed, default is True
        """
        #paths
        if video_name == "" and video_path == "":
            print("Error: saveVideo(video_name, video_path) called without parameters, expected either video_name or video_path")
            return
        if video_name == "":
            video_name = os.path.splitext(os.path.basename(self.videopath))[0]
        else:
            video_path = os.path.join(self.videopath, video_name + ".mp4")
        json_path = os.path.join(self.outpath, video_name + "_tracked.json")
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found, try calling track() first")
            return
        outpath = os.path.join(self.outpath, video_name + "_tracked.mp4")
        #log
        if log is True:
            print(f"saving video {video_name}.mp4 to {self.videopath}", end = "\r")
        #input video
        video = cv2.VideoCapture(video_path)
        #output video
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        box_color = (255, 0, 0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
        #json
        json_data = self.__readFromJson(json_path)
        #iterate over frames, draw bboxes and labels and write to outputfile
        for frame_number, bboxes in enumerate(json_data["labels"]):
            ret, frame = video.read()
            if not ret:
                break
            for bbox in bboxes:
                center_x, center_y, w, h = int(bbox["center_x"] * width), int(bbox["center_y"] * height), int(bbox["w"] * width), int(bbox["h"] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                label = str(bbox["id"])
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            out.write(frame)
                
        video.release()
        out.release()
        
        if compress is True:
            compressed_file_path = os.path.join(self.outpath, video_name + "_tracked_c.mp4")
            subprocess.call(f"ffmpeg -i {outpath} -s 1280x720 -acodec copy -y {compressed_file_path}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(outpath)
            os.rename(compressed_file_path, outpath)
            #vscode can't download/open the file without the next 2 lines
            open_for_vscode_bugfix = cv2.VideoCapture(outpath)
            open_for_vscode_bugfix.release()
            
        #log
        if log is True:
            print(f"video {video_name}.mp4 successfully saved to {self.videopath}")
    
    def __readFromJson(self, path):
        """
        reads data from json file
        parameter:
            path: path to json file
        return value:
            data: data from json file
        """
        with open(path, "r") as file:
            data = json.load(file)
        return data
    
    def __writeToJson(self, path, data, negatives):
        """
        writes bboxes and IDs to json file
        parameter:
            path: path to json file
            data: "labels" part with bboxes including ids for each frame
            negatives: list of IDs for each tracked ID, which can be used as negatives in tripletloss
        """        
        id_data = {"tracked_IDs": [{"id": i, "negatives": list(s)} for i, s in enumerate(negatives)]}
        data = {**id_data, **data}
        with open(path, "w") as file:
            json.dump(data, file, indent=4)

    def __bboxesOverlap(self, bbox1, bbox2, allowed_overlap = 0.25, width = 1920, height = 1080):
        """
        check if bboxes overlap
        parameter:
            bbox1/bbox2: bboxes to check for
            allowed_overlap: % of the area of bboxes which is allowed to overlap between two individuals for still returning false, default is 0.25
            width/height: width and height of video, default is 1920*1080
        return value:
            overlap: boolean; True if bboxes overlap
        """
        resize = 1 - allowed_overlap
        w1, h1, w2, h2 = int(bbox1["w"] * resize * width), int(bbox1["h"] * resize * height), int(bbox2["w"] * resize * width), int(bbox2["h"] * resize * height)
        x1 = int(bbox1["center_x"] * width - w1 / 2)
        y1 = int(bbox1["center_y"] * height - h1 / 2)
        x2 = int(bbox2["center_x"] * width - w2 / 2)
        y2 = int(bbox2["center_y"] * height - h2 / 2)
        overlap = not (x1 + w1 <= x2 or
                    x2 + w2 <= x1 or
                    y1 + h1 <= y2 or
                    y2 + h2 <= y1)
        return overlap

    def __trackIDs(self, data):
        """
        track individuals and create IDs; also removes bboxes when they overlap too much
        parameter:
            data: json data with bboxes for each frame of a video
        return values:
            data: json data of the video including IDs in each bbox
            id_count: int; number of tracked individuals
        """
        id_count = -1
        openIDs = []
        for frame, frame_data in enumerate(data["labels"]):
            bboxes = [bbox for bbox in frame_data if bbox["class"] == 0]
            #iterate over bounding boxes and delete colliding ones
            for bbox in bboxes[:]:
                for otherbbox in bboxes[:]:
                    if bbox != otherbbox and self.__bboxesOverlap(bbox, otherbbox, allowed_overlap = self.allowed_overlap):
                        if bbox in frame_data:
                            frame_data.remove(bbox)
                        if otherbbox in frame_data:
                            frame_data.remove(otherbbox)
            #iterate over remaning bboxes and give IDs
            for bbox in bboxes:
                #check if individual already detected
                for id in openIDs:
                    if self.__bboxesOverlap(bbox, id, allowed_overlap = 0.5) and ((0.9 * id["w"] <= bbox["w"] <= 1.1 * id["w"]) or (0.95 * id["h"] <= bbox["h"] <= 1.05 * id["h"])):
                        bbox["id"] = id["id"]
                        id["ttl"] += 1
                        id["center_x"], id["center_y"], id["w"], id["h"] = bbox["center_x"], bbox["center_y"], bbox["w"], bbox["h"]
                        break
                #new individual
                if "id" not in bbox:
                    id_count += 1
                    bbox["id"] = id_count
                    openIDs.append(dict(id = id_count, center_x = bbox["center_x"], center_y = bbox["center_y"], w = bbox["w"], h = bbox["h"], ttl = 15))
            #decrease ttl (and remove) not tracked IDs from list
            for id in openIDs:
                id["ttl"] -= 1
            openIDs = [openID for openID in openIDs if openID["ttl"] > 0]
            
        return data, id_count

    def __labelFaces(self, data):
        """
        label the face bboxes according to the already labeled body bboxes
        parameter:
            data: json data with bboxes and body IDs for each frame of a video
        return value:
            data: json data of the video including IDs for the face bboxes
        """
        for frame_data in data["labels"]:
            body_bboxes = [bbox for bbox in frame_data if bbox["class"] == 0]
            face_bboxes = [bbox for bbox in frame_data if bbox["class"] == 1]
            for face_bbox in face_bboxes:
                if len(body_bboxes) == 0:
                    frame_data.remove(face_bbox)
                    continue
                for body_bbox in body_bboxes:
                    if self.__bboxesOverlap(face_bbox, body_bbox, allowed_overlap=0):
                        face_bbox["id"] = body_bbox["id"]
                if "id" not in face_bbox:
                    frame_data.remove(face_bbox)
                        
        return data
            
    def __getNegatives(self, data, id_count):
        """
        creates a list of lists which stores IDs which are possible negatives, because they existed at the same time
        parameter:
            data: json data with bboxes and IDs for each frame of a video
        return value:
            negatives: list of lists; at negatives[ID] stores a list of IDs which are negatives for ID
        """
        negatives = [set() for i in range(id_count + 1)]
        for frame, frame_data in enumerate(data["labels"]):
            bboxes = [bbox for bbox in frame_data if bbox["class"] == 0]
            frame_ids = set()
            for bbox in bboxes:
                frame_ids.add(bbox["id"])
            for id in frame_ids:
                for frame_id in frame_ids:
                    if frame_id != id:
                        negatives[id].add(frame_id)
        return negatives
    