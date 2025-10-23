# artificial-vision

This project enables the detection and tracking of pedestrians for attribute recognition, including gender, upper color, lower color, presence of a bag, and presence of a hat. Additionally, the project counts the number of passages and time for each tracked individual within two possible regions. The file contains information about each tracked individual, including their id. The detector used is Yolo8s, while the tracker used is Deep Sort.

This project participated in the Artificial Vision contest launched by the MIVIA Lab of the University of Salerno in 2023/2024 and aims to solve the challenges currently present in Pedestrian Attribute Recognition(PAR). The following project was developed by a team of three people: Carlo Santonicola, Lorenzo Mignone, and MariaPia Lombardi.
The different PAR challenges faced in this project for this contest:
• Multi-views: The images taken from different angles by the camera lead to the viewpoint issues for many computer vision tasks. Since the human body is not rigid, the person attribute recognition is more complicated.
• Occlusion: Partial occlusion of human body by other person or things increases the difficulty of person attributes recognition.
• Low resolution: In practical scenarios, the resolution of images are rather low since people may be framed at a long distance.
• Illumination: the light condition is variable at different time. The shadow may also be taken in the person images and the images taken from nighttime maybe totally ineffective.
• Blur: Recognizing attributes in this situation is also a very challenging task.
• Unbalanced attribute distribution: The number and classes of available attributes are variable in the datasets.

# Dependencies

Before starting to use the project, ensure that your system satisfies the following dependencies by executing the following command in your terminal:

```bash
pip3 install -r requirements.txt
```

Then, download PAR model from the following link and move it to the models folder:

https://drive.google.com/drive/folders/1ZVa2NCow_KgHzf7G2JtQiUMmdbO2jToP?usp=drive_link

# Run project

Execute the project by running this command. You must specify your video, your own  `config.txt` for roi coords and eventually the `results.txt` . 

```bash
python3 group04.py --video video.mp4 -c config.txt -r results.txt
```
