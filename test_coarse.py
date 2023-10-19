import simulation
import numpy as np
import time
import csv
import complex_assignment
import sys
import math

def reshape_to_3d(matrix_2d, width, height, interval):
    matrix_3d = [[[0]*interval for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            for k in range(interval):
                matrix_3d[i][j][k] = matrix_2d[i * width + j][k]
    return matrix_3d

def reshape_to_2d(matrix_3d):
    height, width, interval = len(matrix_3d), len(matrix_3d[0]), len(matrix_3d[0][0])
    matrix_2d = [matrix_3d[i//width][i%width] for i in range(height*width)]
    return matrix_2d

# Convert a 2d index to a 1d index with wrapping
def whtoi(x: int, y: int, width: int, height: int) -> int:
    if x < 0:
        x += width
    if x >= width:
        x -= width
    if y < 0:
        y += height
    if y >= height:
        y -= height
    return y * width + x

def coarsen(matrix_2d, width, height, interval,coarse_constant):
    # Convert 2D matrix to 3D
    matrix_3d = reshape_to_3d(matrix_2d, width, height, interval)
    
    # Calculate the coarsened dimensions
    coarse_width = math.ceil(width/coarse_constant)
    coarse_height = math.ceil(height/coarse_constant)
    print(coarse_width,coarse_height)
    
    coarse_matrix = [[[0]*interval for _ in range(coarse_height)] for _ in range(coarse_width)]

    for y in range(height):
        j = math.floor(y/coarse_constant)
        for x in range(width):
            i = math.floor(x/coarse_constant)
            for k in range(interval):
                print("k: ",k)
                # coarse_matrix[i][j][k] += matrix_2d[whtoi(x, y, width, height)][k]
                coarse_matrix[i][j][k] += matrix_3d[x][y][k]

    # Convert coarsened 3D matrix back to 2D
    coarse_matrix_2d = reshape_to_2d(coarse_matrix)

    return coarse_matrix_2d

def main():
    width = 4
    height = 4
    intervals = 1
    file_path = 'random_workload.csv'
    # with open(file_path, 'r') as file:
    #     csv_reader = csv.reader(file)
    #     width = sum(1 for row in csv_reader)
    samples = width * height
    # read into the workload matrix and update interval
    workload_matrix = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=' ')
        for row in csv_reader:
            if len(row) > 0:
                row = [int(element) for element in row]
                intervals = len(row)
                workload_matrix.append(row)
    print("original workload: \n",workload_matrix)
    print("processed workload: \n",coarsen(workload_matrix,width,height,intervals,2))
    # tmp = reshape_to_3d(workload_matrix, width, height, intervals)
    # print("processed workload: \n",tmp)
    # print("processed workload: \n",reshape_to_2d(tmp))
if __name__ == '__main__':
    main()