from utils.dataset import get_vertices, get_edges, format_edges, transform_geom
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True, help='Folder name whose data must be processed.')
ARGS, unknown = parser.parse_known_args()

if __name__ == '__main__':
    vertices, obj_labels, obj_ids = get_vertices(folder=ARGS.folder, file_name='nodes')  # (N, 3)
    edges = get_edges(folder=ARGS.folder, file_name='edges')  # (M, 4)
    # centers = get_centers(labels=labels, vertices=vertices, file_name='centers')  # (C, 2)
    edges_form = format_edges(vertices, edges, save=True, folder=ARGS.folder, name='edges_form')
    # transform
    vertices_form = transform_geom(points=vertices, coord_reference=vertices, folder=ARGS.folder, name='vertices_norm')
