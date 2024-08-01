from utilities import video_read, video_write
from tracking import Tracker
from coloring import TeamAssigner
import os

def main():
    while True:
        input_pth = input("Enter your input file's path: ")
        if os.path.exists(input_pth):
            break
        print("Provided path does not exist. Enter a different video file path.")
    output_name = input("Enter your output file's name: ")
    output_pth = "output/" + output_name +".avi"
    input_frames = video_read(input_pth)
    tracker = Tracker("models/best.pt")
    tracks = tracker.get_object_track(input_frames, cached=True, cache_path="cache_tracks/tracks.pkl")

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(input_frames[0], tracks['player'][0])
    for frame_number, player_track in enumerate(tracks["player"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(input_frames[frame_number], track["bound"], player_id)
            tracks["player"][frame_number][player_id]['team'] = team
            tracks["player"][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    output_frames = tracker.draw_annotations(tracks, input_frames)
    video_write(output_frames, output_pth)

if __name__ == '__main__':
    main()
