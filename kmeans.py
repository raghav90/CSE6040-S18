from PIL import Image
import numpy as np
import random
import time


def read_img(path):
	"""
	Read image and store it as an array, given the image path. 
	Returns the 3 dimensional image array.
	"""
	img = Image.open(path)
	img_arr = np.array(img, dtype='int32')
	img.close()
	return img_arr

def img_reshape(img_arr, order="C"):
	"""
	Reshape the image from a 3-D (RGB) to a 2-D array.
	input : 3 dimensional image array
	output : 2 dimensional, "flattened" array
	"""
	r, c, l = img_arr.shape
	img_arr_reshape = np.reshape(img_arr, (r*c, l), order=order)
	return img_arr_reshape

def display_image(arr):
	"""
	display and save the image
	input : 3 dimensional array
	"""
	arr = arr.astype(dtype='uint8')
	img = Image.fromarray(arr, 'RGB')
	img.show()
	img.save('my_image.jpg')

def read_image_rgb(img_bmp, order="C"):
	img_reshaped = img_reshape(img_bmp)
	return img_reshaped

def init_centroids(arr, k):
	"""
	Initializes the centroids.
	inputs: 
	arr : Input array (2 dimensional "flattened")
	k : The number of clusters in the problem
	output: Coordinates of the centroids
	"""
	r, c = arr.shape
	centroids = random.sample(range(r), k)
	init_points = arr[centroids]
	m = np.unique(init_points)
	if m.shape[0] < init_points.shape[0]:
		init_centroids(arr, k)
	else:
		return arr[centroids]

def loss_fn(dist):
	"""
	Returns the mean distance (squared) from all points to the cluster centers
	"""
	J = np.min(dist, axis=0)
	return np.mean(J)

def centroid_distance(arr, centroid, k):
	"""
	Measures the distance from centroid to all other points in the data array
	"""
	dist = []
	for i in range(k):
		dist_vec = arr - centroid[i]
		dist_vec_norm = np.sum(abs(dist_vec)**2, axis=-1)
		dist.append(dist_vec_norm)
	return np.array(dist)

def new_clusters(img_arr, clusters_index, k):
	"""
	Assign data points to their clusters
	"""
	clusters = []
	for i in range(k):
		ind = np.where(clusters_index==i)
		members = img_arr[ind]
		clusters.append(members)
	return clusters

def get_centroids(clusters):
	"""
	Get centroids for the clusters (cluster centers)
	"""
	centroids = [np.mean(c, axis=0) for c in clusters]
	return np.array(centroids)

def iterate_kmeans(img_arr, initial_centroids, k, stop_delta = .001, max_iterate=200):
	"""
	Iterate kmeans till convergence.
	inputs
	img_arr : flattened the image
	initial_centroids : randomly initialized centroids
	k : number of clusters
	stop_delta : convergence criterion
	max_iterat : maximum number of iterations till convergence
	returns : The final cluster index for all points and the centroid coordinates
	"""
	energy = 0
	delta = 100000
	count = 0
	while(delta>stop_delta):
		count += 1
		if count >= max_iterate:
			break
		else:
			dist = centroid_distance(img_arr, initial_centroids, k)
			J = loss_fn(dist)
			delta = abs(energy - J)
			energy = J
			clusters_index = np.argmin(dist, axis=0)
			clusters = new_clusters(img_arr, clusters_index, k)
			centroids = get_centroids(clusters)
			initial_centroids = centroids
	return clusters_index, np.rint(centroids)

def cluster_image(path, k):
	"""
	Apply clustering to the image.
	Input:
	path : The path to the image
	k : Number of clusters
	Returns :
	Displays the clustered image and saves the clustered image in the current directory
	"""
	img_bmp = read_img(path)
	r, c, l = img_bmp.shape
	img_arr = read_image_rgb(img_bmp)
	initial_centroids = init_centroids(img_arr, k)
	clusters_index, centroid = iterate_kmeans(img_arr, initial_centroids, k)
	img_new = np.array([centroid[i] for i in clusters_index])
	img2 = np.reshape(img_new, (r, c, l), order="C")
	display_image(img2)

if __name__ == "__main__":
	path = "C:\\Users\\Raghav\\Downloads\\testfaces.jpg"
	cluster_image(path, 3)