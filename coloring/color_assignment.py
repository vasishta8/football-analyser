from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_to_team = {}
        pass

    def get_clustering(self, image):
        image2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image2d)
        return kmeans

    def get_player_color(self, frame, bound):
        image = frame[int(bound[1]):int(bound[3]), int(bound[0]):int(bound[2])]
        top_half = image[0:int((image.shape[0]/2)),:]
        clusters = self.get_clustering(top_half)
        labels = clusters.labels_
        clustered_img = labels.reshape(top_half.shape[0], top_half.shape[1])
        corners = [clustered_img[0,0], clustered_img[0,-1], clustered_img[-1,0], clustered_img[-1,-1]]
        background_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 if background_cluster == 0 else 0
        player_color = clusters.cluster_centers_[player_cluster]
        return player_color


    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bound = player_detection["bound"]
            player_color = self.get_player_color(frame, bound)
            player_colors.append(player_color)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, bound, player_id):
        if player_id in self.player_to_team:
            return self.player_to_team[player_id]
        player_color = self.get_player_color(frame, bound)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1
        self.player_to_team[player_id] = team_id
        return team_id
