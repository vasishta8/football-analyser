# Football Analyser

A football anaylser that can detect and track the players, referees and the ball. Utilises YOLO for detection, and Supervision for tracking. It also uses the K-means clustering method from the scikit-learn library to segment and obtain the player's jersey colour - to identify their team.

To run it, download or clone the repo, and run the training.ipynb file. Download or move the best.pt model in the "training/runs/detect/train/weights" folder, and place it in the "models" folder. Then, run main.py
