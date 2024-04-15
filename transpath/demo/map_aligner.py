import argparse

class MapAligner:
    def __init__(self, large_map_points, small_map_points):
        self.large_map_points = large_map_points
        self.small_map_points = small_map_points
        self.scaling_factor_x, self.scaling_factor_y = self.calculate_scaling_factors()
        self.translation_x, self.translation_y = self.calculate_translation()

    def calculate_scaling_factors(self):
        # Calculate scaling factors based on the distances between the two points in each map
        scale_x = (self.small_map_points[1][0] - self.small_map_points[0][0]) / \
                  (self.large_map_points[1][0] - self.large_map_points[0][0])
        scale_y = (self.small_map_points[1][1] - self.small_map_points[0][1]) / \
                  (self.large_map_points[1][1] - self.large_map_points[0][1])
        return scale_x, scale_y

    def calculate_translation(self):
        # Adjust translation based on the first point after scaling
        trans_x = self.small_map_points[0][0] - (self.large_map_points[0][0] * self.scaling_factor_x)
        trans_y = self.small_map_points[0][1] - (self.large_map_points[0][1] * self.scaling_factor_y)
        return trans_x, trans_y

    def translate_coordinates(self, large_map_coord):
        # Apply scaling and then translation
        x, y = large_map_coord
        x_small = (x * self.scaling_factor_x) + self.translation_x
        y_small = (y * self.scaling_factor_y) + self.translation_y
        return x_small, y_small

def parse_points(point_str):
    points = [tuple(map(float, p.split(','))) for p in point_str.split(';')]
    return points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Map Aligner Script')
    parser.add_argument('--large_map_points', type=str, required=False, default='-10,-1670;3450,200',
                        help='Two points from the larger map in the format "x1,y1;x2,y2"')
    parser.add_argument('--small_map_points', type=str, required=False, default='58,220;166,8',
                        help='Two corresponding points from the smaller map in the format "x1,y1;x2,y2"')
    parser.add_argument('--translate_point', type=str, required=False, default='2040,-210',
                        help='Point from the larger map to translate in the format "x,y"')

    args = parser.parse_args()

    large_map_points = parse_points(args.large_map_points)
    small_map_points = parse_points(args.small_map_points)
    translate_point = tuple(map(float, args.translate_point.split(',')))

    aligner = MapAligner(large_map_points, small_map_points)
    translated_coord = aligner.translate_coordinates(translate_point)

    print(f"Translated coordinates: {translated_coord}")
