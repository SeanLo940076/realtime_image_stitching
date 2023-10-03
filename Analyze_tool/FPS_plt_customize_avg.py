import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# XML file path
xml_file_path = "/home/sean/Desktop/stitching_realtime/build/FPS_data.xml"
sampling_rate = 100 # 每100點做一次紀錄
x_axis_spacing = 1000
y_axis_spacing = 2

# Check if the file exists
if os.path.exists(xml_file_path):
    # Parse the XML data from file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract the frame IDs and FPS values
    frame_ids = []
    fps_values = []

    for fps_value in root.findall("FPS_value"):
        frame_ids.append(int(fps_value.get("id")))
        fps_values.append(float(fps_value.text))

    # Calculate the average for every nth frame
    frame_ids_avg = []
    fps_values_avg = []
    for i in range(0, len(frame_ids), sampling_rate):
        frame_ids_avg.append(frame_ids[i])
        fps_values_avg.append(sum(fps_values[i:i+sampling_rate]) / sampling_rate)

    # Create the plot
    plt.figure(figsize=(15, 10))
    plt.plot(frame_ids_avg, fps_values_avg, marker='o')
    plt.title("Frame Rate Over Time")
    plt.xlabel("Frame ID (frames)")
    plt.ylabel("FPS (frames per second)")
    plt.grid(True)

    # Set x-axis and y-axis ticks based on the data range
    plt.xticks(range(0, max(frame_ids_avg) + x_axis_spacing, x_axis_spacing))  # Set x-axis ticks to be every 1000 frames
    plt.yticks(range(0, int(max(fps_values_avg)) + y_axis_spacing, y_axis_spacing))   # Set y-axis ticks to be every 2 FPS

    # Save the figure as a .jpg file with 600 dpi
    plt.savefig("/home/sean/Desktop/stitching_realtime/FPS_plot_" + str(sampling_rate) + "_avg.jpg", dpi = 600)
    print("Save in /home/sean/Desktop/stitching_realtime/FPS_plot_" + str(sampling_rate) + "_avg.jpg")
else:
    print(f"File {xml_file_path} does not exist.")
