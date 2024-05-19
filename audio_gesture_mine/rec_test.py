import pyaudio
import numpy as np
import cv2
import mediapipe as mp
import os
import deeplake
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data", default="/Users/nikny/local_data/deep_lake_ds", type=str)
parser.add_argument("--audio_path", default="/Users/nikny/local_data/marginal_audio_tensors", type=str) 
args = parser.parse_args()

class RecData:
    def __init__(self, data_path, audio_path):
        self.local_path = data_path
        self.marginal_audio_path = audio_path

    @staticmethod
    def load_numpy_data(path):
        """Load numpy array from file."""
        return np.load(path)

    @staticmethod
    def get_files(data_dir):
        """Retrieve audio and motion file paths indexed by video ID."""
        files = os.listdir(data_dir)
        audio_files = {}
        motion_files = {}
        for file in files:
            if file.endswith("_audio.npy"):
                video_id = file.split("_audio.npy")[0]
                audio_files[video_id] = os.path.join(data_dir, file)
            elif file.endswith("_mocap.npy"):
                video_id = file.split("_mocap.npy")[0]
                motion_files[video_id] = os.path.join(data_dir, file)
        return audio_files, motion_files
    
    def get_marginal_audio(self, nr_audio_buffers):
        """Retrieve marginal audio data to reach the required number of audio buffers."""
        marginal_audio = [d for d in os.listdir(self.marginal_audio_path) if d.endswith('.npy')]
        marginal_audio_data = np.array([])
        while len(marginal_audio_data) < nr_audio_buffers:
            marginal_audio_id = np.random.choice(marginal_audio)
            marginal_audio_path = os.path.join(self.marginal_audio_path, marginal_audio_id)
            marginal_audio_additional = self.load_numpy_data(marginal_audio_path)
            if marginal_audio_data.size != 0:
                marginal_audio_data = np.concatenate((marginal_audio_data, marginal_audio_additional))
            else:
                marginal_audio_data = marginal_audio_additional
        return marginal_audio_data[:nr_audio_buffers]

    def preprocess_and_store_in_deeplake(self, audio_data, motion_data):
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        try:
            ds = deeplake.load(self.local_path)
        except Exception:
            ds = deeplake.empty(self.local_path)
            ds.create_tensor('motion', htype='sequence')
            ds.create_tensor('audio', htype='audio', sample_compression=None)
            ds.create_tensor('audio_ahead', htype='audio', sample_compression=None)
            ds.create_tensor('audio_marginal', htype='audio', sample_compression=None)

        num_buffers = audio_data.shape[0] - 1
        marginal_audio_data = self.get_marginal_audio(num_buffers + 1)

        motion_to_audio_ratio = motion_data.shape[0] / num_buffers

        # Create a batch dictionary
        batch = {
            'motion': [],
            'audio': [],
            'audio_ahead': [],
            'audio_marginal': []
        }

        for i in range(num_buffers):
            audio_chunk = audio_data[i, :]
            motion_index = int(i * motion_to_audio_ratio)
            motion_chunk = motion_data[motion_index, :, :]

            batch['audio'].append(audio_chunk)
            batch['motion'].append(motion_chunk)
            ahead_idx = (i + 1) % num_buffers
            batch['audio_ahead'].append(audio_data[ahead_idx, :])
            batch['audio_marginal'].append(marginal_audio_data[i, :])

        # Convert lists to numpy arrays and add them as a new batch
        ds.append({
            'motion': np.array(batch['motion']),
            'audio': np.array(batch['audio']),
            'audio_ahead': np.array(batch['audio_ahead']),
            'audio_marginal': np.array(batch['audio_marginal'])
        })

    def rec(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 2048
        DEVICE_INDEX = 1

        p = pyaudio.PyAudio()
        saved_audio_data = []
        saved_poses = []

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()

        def audio_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            saved_audio_data.append(audio_data)
            return (None, pyaudio.paContinue)

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=DEVICE_INDEX,
                        frames_per_buffer=CHUNK,
                        stream_callback=audio_callback)
        stream.start_stream()

        cap = cv2.VideoCapture(0)
        print("Recording... Press 'q' to stop.")
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                if results.pose_landmarks:
                    pose_landmarks = results.pose_landmarks.landmark
                    pose_data = [[landmark.x, landmark.y, landmark.z] for landmark in pose_landmarks]
                    saved_poses.append(pose_data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            cap.release()
            stream.stop_stream()
            stream.close()
            p.terminate()
            cv2.destroyAllWindows()

            audio_data = np.array(saved_audio_data, dtype=np.float32)
            motion_data = np.array(saved_poses, dtype=np.float32)
            self.preprocess_and_store_in_deeplake(audio_data, motion_data)

if __name__ == '__main__':
    app = RecData(args.data, args.audio_path)
    app.rec()
