import tempfile
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect


def main():

    st.set_page_config(page_title="Football Tactical Analysis Using Computer Vision",
                       layout="wide", initial_sidebar_state="collapsed")
    st.title("Dynamic Tactical Video Analysis and Visualization of Football Match Events Using Computer Vision and  Data Analytics")
    st.subheader(":blue[Optimized for Tactical Camera Footage]")

    st.sidebar.title("Side Bar Panel")
    demo_selected = st.sidebar.radio(label="Select Demo Video", options=[
                                     "Demo 1", "Demo 2"], horizontal=True)

    # Sidebar Setup
    st.sidebar.markdown('---')
    st.sidebar.subheader("Video Upload")
    input_vide_file = st.sidebar.file_uploader(
        'Upload a video file', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])

    demo_vid_paths = {
        "Demo 1": './demo_vid_1.mp4',
        "Demo 2": './demo_vid_2.mp4'
    }
    demo_vid_path = demo_vid_paths[demo_selected]
    demo_team_info = {
        "Demo 1": {"team1_name": "France",
                   "team2_name": "Switzerland",
                   "team1_p_color": '#003f5c',
                   "team1_gk_color": '#ffa600',
                   "team2_p_color": '#bc5090',
                   "team2_gk_color": '#ff6361',
                   },
        "Demo 2": {"team1_name": "Chelsea",
                   "team2_name": "Manchester City",
                   "team1_p_color": '#2f4b7c',
                   "team1_gk_color": '#f95d6a',
                   "team2_p_color": '#665191',
                   "team2_gk_color": '#ffa600',
                   }
    }
    selected_team_info = demo_team_info[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Demo Video')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Uploaded Video')
        st.sidebar.video(demo_bytes)

    # Load the YOLOv8 players detection model
    model_players = YOLO("../models/PlayersModel/weights/best.pt")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("../models/FiieldModel/weights/best.pt")

    st.sidebar.markdown('---')
    st.sidebar.subheader("Teams")
    team1_name = st.sidebar.text_input(
        label='Team 1 Name', value=selected_team_info["team1_name"])
    team2_name = st.sidebar.text_input(
        label='Team 2 Name', value=selected_team_info["team2_name"])
    st.sidebar.markdown('---')

    # Page Setup
    tab1, tab2, tab3 = st.tabs(
        ["Instructions", "Team Colors", "Settings & Detection"])

    with tab1:
        st.header(':orange[Getting Started]')
        st.subheader('Main Features:', divider='orange')
        st.markdown("""
                    1. Detection of players, referees, and the ball.
                    2. Team identification and prediction.
                    3. Tactical positioning and mapping.
                    4. Ball tracking across the field.
                    """)
        st.subheader('Steps to Use:', divider='orange')
        st.markdown("""
                    **Demo videos with recommended settings are preloaded. Follow these steps to analyze your own video:**
                    1. Upload a video using the "Upload Video" option in the sidebar.
                    2. Enter the team names for the video in the sidebar.
                    3. Navigate to the "Team Colors" tab.
                    4. Select a frame where all players and goalkeepers are visible.
                    5. Pick each team's colors following the on-screen instructions.
                    6. Move to the "Settings & Detection" tab, adjust parameters if necessary, and run the analysis.
                    7. The output video, if saved, will be located in the "outputs" directory.
                    """)

    with tab2:
        t1col1, t1col2 = st.columns([1, 1])
        with t1col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select Frame", min_value=1, max_value=frame_count,
                                  step=1, help="Choose a frame to extract team colors.")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            with st.spinner('Detecting players in the selected frame...'):
                results = model_players(frame, conf=0.7)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((80, 60, 3), dtype=np.uint8) * 255
                for i, j in enumerate(list(labels)):
                    if int(j) == 0:
                        bbox = bboxes[i, :]
                        obj_img = frame_rgb[int(bbox[1]):int(
                            bbox[3]), int(bbox[0]):int(bbox[2])]
                        obj_img = cv2.resize(obj_img, (60, 80))
                        detections_imgs_list.append(obj_img)
                detections_imgs_grid.append(
                    [detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(
                    len(detections_imgs_list)//2, len(detections_imgs_list))])
                if len(detections_imgs_list) % 2 != 0:
                    detections_imgs_grid[0].append(padding_img)
                concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                concat_det_imgs = cv2.vconcat(
                    [concat_det_imgs_row1, concat_det_imgs_row2])
            st.write("Detected Players")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
            st.markdown('---')
            radio_options = [f"{team1_name} Player Color", f"{team1_name} Goalkeeper Color",
                             f"{team2_name} Player Color", f"{team2_name} Goalkeeper Color"]
            active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options, horizontal=False,
                                    help="Choose the color to pick and click on the image above. Adjust colors below.")
            if value is not None:
                picked_color = concat_det_imgs[value['y'], value['x'], :]
                st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(
                    picked_color)
            st.write("Use the color pickers below to fine-tune colors.")
            cp1, cp2, cp3, cp4 = st.columns([1, 1, 1, 1])
            with cp1:
                hex_color_1 = st.session_state.get(
                    f"{team1_name} Player Color", selected_team_info["team1_p_color"])
                team1_p_color = st.color_picker(
                    label='Team 1 Player', value=hex_color_1, key='t1p')
                st.session_state[f"{team1_name} Player Color"] = team1_p_color
            with cp2:
                hex_color_2 = st.session_state.get(
                    f"{team1_name} Goalkeeper Color", selected_team_info["team1_gk_color"])
                team1_gk_color = st.color_picker(
                    label='Team 1 Goalkeeper', value=hex_color_2, key='t1gk')
                st.session_state[f"{team1_name} Goalkeeper Color"] = team1_gk_color
            with cp3:
                hex_color_3 = st.session_state.get(
                    f"{team2_name} Player Color", selected_team_info["team2_p_color"])
                team2_p_color = st.color_picker(
                    label='Team 2 Player', value=hex_color_3, key='t2p')
                st.session_state[f"{team2_name} Player Color"] = team2_p_color
            with cp4:
                hex_color_4 = st.session_state.get(
                    f"{team2_name} Goalkeeper Color", selected_team_info["team2_gk_color"])
                team2_gk_color = st.color_picker(
                    label='Team 2 Goalkeeper', value=hex_color_4, key='t2gk')
                st.session_state[f"{team2_name} Goalkeeper Color"] = team2_gk_color
        st.markdown('---')

        with t1col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR")

    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state[f"{team1_name} Player Color"], st.session_state[f"{team1_name} Goalkeeper Color"],
                                                    team2_name, st.session_state[f"{team2_name} Player Color"], st.session_state[f"{team2_name} Goalkeeper Color"])

    with tab3:
        t2col1, t2col2 = st.columns([1, 1])
        with t2col1:
            player_model_conf_thresh = st.slider(
                'Players Detection Confidence', min_value=0.0, max_value=1.0, value=0.65)
            keypoints_model_conf_thresh = st.slider(
                'Field Keypoints Detection Confidence', min_value=0.0, max_value=1.0, value=0.75)
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement Tolerance (pixels)', min_value=-1, max_value=100, value=5,
                                                        help="Max allowed average distance between the field keypoints in consecutive frames.")
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Palette Colors", min_value=1, max_value=5, step=1, value=3,
                                       help="Number of colors to extract from detected players for team prediction.")
            st.markdown("---")
            save_output = st.checkbox(label='Save Output Video', value=False)
            if save_output:
                output_file_name = st.text_input(
                    label='Output File Name (Optional)', placeholder='Enter video file name.')
            else:
                output_file_name = None
        st.markdown("---")

        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            nbr_frames_no_ball_thresh = st.number_input("Ball Track Reset Threshold (frames)", min_value=1, max_value=10000,
                                                        value=30, help="Number of frames with no ball detection before resetting track.")
            ball_track_dist_thresh = st.number_input("Ball Track Distance Threshold (pixels)", min_value=1, max_value=1280,
                                                     value=120, help="Max distance between consecutive ball detections for tracking.")
            max_track_length = st.number_input("Maximum Track Length (detections)", min_value=1, max_value=1000,
                                               value=40, help="Max number of ball detections to keep in tracking history.")
            ball_track_hyperparams = {
                0: nbr_frames_no_ball_thresh,
                1: ball_track_dist_thresh,
                2: max_track_length
            }
        with bcol2:
            st.write("Annotation Options:")
            bcol21t, bcol22t = st.columns([1, 1])
            with bcol21t:
                show_k = st.toggle(label="Show Keypoints", value=False)
                show_p = st.toggle(label="Show Players", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Show Color Palettes", value=True)
                show_b = st.toggle(label="Show Ball Tracks", value=True)
            plot_hyperparams = {
                0: show_k,
                1: show_pal,
                2: show_b,
                3: show_p
            }
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 3, 3, 1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = not (team1_name and team2_name)
                start_detection = st.button(
                    label='Run Detection', disabled=ready)
            with bcol23:
                stop_btn_state = not start_detection
                stop_detection = st.button(
                    label='Stop Detection', disabled=stop_btn_state)
            with bcol24:
                st.write('')

    stframe = st.empty()
    cap = cv2.VideoCapture(tempf.name)
    status = False

    if start_detection and not stop_detection:
        st.toast(f'Detection Running...')
        status = detect(cap, stframe, output_file_name, save_output, model_players, model_keypoints,
                        detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
                        num_pal_colors, colors_dic, color_list_lab)
    else:
        try:
            cap.release()
        except:
            pass
    if status:
        st.toast(f'Detection Completed!')
        cap.release()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
