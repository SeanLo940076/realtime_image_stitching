import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# XML file path
xml_file_path = "/home/sean/Desktop/stitching_realtime/build/FPS_data.xml"
# xml_file_path = "/home/sean/Desktop/stitching_realtime/build/imagegenerate_data.xml"
# xml_file_path = "/home/sean/Desktop/stitching_realtime/build/YOLOtime_data.xml"
sampling_rate = 250 # 每100點做一次紀錄
x_axis_spacing = 3000
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

    # Select every nth frame
    frame_ids_number = frame_ids[::sampling_rate]
    fps_values_number = fps_values[::sampling_rate]

    # Create the plot
    plt.figure(figsize=(15, 10))
    plt.plot(frame_ids_number, fps_values_number, marker='o')
    plt.title("Frame Rate Over Time")
    plt.xlabel("Frame ID (frames)")
    plt.ylabel("FPS (frames per second)")
    plt.grid(True)

    # Set x-axis and y-axis ticks based on the data range
    plt.xticks(range(0, max(frame_ids_number) + x_axis_spacing, x_axis_spacing))  # Set x-axis ticks to be every 1000 frames
    plt.yticks(range(0, int(max(fps_values_number)) + y_axis_spacing, y_axis_spacing))   # Set y-axis ticks to be every 2 FPS

    # Save the figure as a .jpg file with 600 dpi
    plt.savefig("/home/sean/Desktop/stitching_realtime/FPS_plot_" + str(sampling_rate) + ".png", dpi = 600)
    print("Save in /home/sean/Desktop/stitching_realtime/FPS_plot_" + str(sampling_rate) + ".png")
else:
    print(f"File {xml_file_path} does not exist.")
