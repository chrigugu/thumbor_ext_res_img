from PIL import Image
from PIL import ImageFilter
from PIL import ImageStat
from PIL import ImageOps
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
import glob
import ntpath
import math
import numpy as np
from scipy import stats
import subprocess
import os
import re
import commands
import time
import matplotlib.pyplot as plt
from itertools import islice
import cv2
from pyexiv2 import ImageMetadata
import requests


def evaluateWriteImageMemory(fname, loop_amount, use_opencv):
	im = Image.open(fname);
	width = im.size[0];
	height = im.size[1];
	print str(width),"x", str(height); 
	step = int(width/4);

	o_x = [];
	o_y = [];
	time_arr = [];
	for i in xrange(0, loop_amount, 1):
		time_start = time.time();
		if use_opencv:
			writeImageCVMemory(fname, step);
		else:
			writeImageMemory(fname, step);
		time_arr.append(time.time() - time_start);
		# print time.time() - time_start;
	print np.average(time_arr);

	metadata = ImageMetadata(fname);
	metadata.read();
	key = 'Iptc.Application2.Keywords';

	quadratic = float(metadata[key].values[0]);
	linear = float(metadata[key].values[1]);
	intercept = float(metadata[key].values[2]);

	x = [];
	step_regr = int(width/20);
	for i in xrange(step_regr, width, step_regr):
		x.append(i);

	plt.xlabel('image width [pixel]');
	plt.ylabel('file size [bytes]');

	y_d = np.asarray(x)*np.asarray(x)*quadratic + np.asarray(x)*linear + intercept;

	max_file_size = int(max(y_d));
	min_file_size = int(min(y_d));
	range_file_size = max_file_size - min_file_size;

	res_y = [];
	res_x = [];
	res_step = int(range_file_size/10);

	for i in xrange(min_file_size, range_file_size, res_step):
		res_y.append(i);

	res_y.append(max_file_size);
	res_x = (np.sqrt(np.asarray(res_y)*4*quadratic + linear*linear - 4*quadratic*intercept) - linear)/(2*quadratic);
	plt.plot(res_x, res_y, 'o', x, y_d);

def writeImageMemory(img, step):
	# img = Image.open(fname);
	# img.save('./test.jpg', format='JPEG', subsampling = 0);
	width = img.size[0];
	height = img.size[1];
	x = [];
	y = [];

	for i in xrange(step, width, step):
		r = float(i) / float(width);
		height_r = int(height * r);
		dim = (i, height_r);
		pil_resized = img.resize(dim, Image.LANCZOS); # .BILINEAR .LANCZOS .ANTIALIAS .NEAREST .BICUBIC
		out = BytesIO();
		# pil_resized.save(out,format = 'JPEG', optimize = 0, qtables="keep", subsampling = -1, quality = compression_quality); # subsampling -1 -> keep, 0 -> 4:4:4, 1 -> 4:2:2, 2 -> 4:1:1
		pil_resized.save(out, format = 'JPEG', subsampling = -1, optimize = False, progressive = True, quality = compression_quality); # subsampling -1 -> keep, 0 -> 4:4:4, 1 -> 4:2:2, 2 -> 4:1:1
		# pil_resized.save(out, format = 'JPEG'); # subsampling -1 -> keep, 0 -> 4:4:4, 1 -> 4:2:2, 2 -> 4:1:1
		out.seek(0, os.SEEK_END);
		file_size = out.tell();
		y.append(file_size);
		x.append(i);
		# f_sampling.write('[%d, %d, "%s", %d],\n' %(i, file_size, ntpath.basename(fname), compression_quality));

	plt.xlabel('image width [pixel]');
	plt.ylabel('file size [bytes]');
	plt.plot(x, y);

	# quadratic, linear, intercept = np.polyfit(x, y, 2);
	# metadata = ImageMetadata(fname);
	# metadata.read();
	# key = 'Iptc.Application2.Keywords';
	# metadata[key] = [str(quadratic), str(linear), str(intercept)];
	# metadata.write();

def writeImageCVMemory(pil_image, step):
	img = np.array(pil_image)
	height, width, channels = img.shape
	x = [];
	y = [];

	for i in xrange(step, width, step):
		r = float(i) / float(width);
		height_r = int(height * r);
		dim = (i, height_r);
		resized = cv2.resize(img, dim, 0.5, 0.5, interpolation = cv2.INTER_LANCZOS4); # .INTER_AREA .INTER_LANCZOS4 .INTER_NEAREST .INTER_CUBIC

		# resized = cv2.resize(img, dim, 1.0, 1.0, interpolation = cv2.INTER_LANCZOS4); # .INTER_AREA .INTER_LANCZOS4 .INTER_NEAREST .INTER_CUBIC

		# resized = img;

		# transcol=cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
		# SSV = 2;
		# SSH = 2;
		# crf = cv2.boxFilter(transcol[:,:,1], ddepth = -1, ksize = (2, 2));
		# cbf = cv2.boxFilter(transcol[:,:,2], ddepth = -1, ksize = (2, 2));
		# crsub = crf[::SSV, ::SSH];
		# cbsub = cbf[::SSV, ::SSH];
		# # imSub = [transcol[:, :, 0], crsub, cbsub];
		# # imSub = [];
		# # imSub.append(transcol[:, :, 0]);
		# # imSub.append(crsub);
		# # imSub.append(cbsub);

		# # b,g,r = cv2.split(img);

		# # crsub = np.zeros((width,height,1), 'uint8');
		# # cbsub = np.zeros((width,height,1), 'uint8');

		# # imSub = cv2.merge((transcol[:, :, 0], crsub, cbsub));

		# # rgbArray = np.zeros((width,height,3), 'uint8')
		# # rgbArray[..., 0] = transcol[:, :, 0];
		# # rgbArray[..., 1] = crsub;
		# # rgbArray[..., 2] = cbsub;

		# # imSub = np.dstack((transcol[:, :, 0], crsub, cbsub));
		# # imSub[..., 0] = transcol[:, :, 0];
		# # imSub[..., 1] = crsub;
		# # imSub[..., 2] = cbsub;

		# imSub=[transcol[:,:,0],crsub,cbsub];
		# print len(transcol[:,:,0]);
		# print len(crsub);
		# print imSub;
		# cv2.imwrite( "./test.jpg", cv2.cvtColor(transcol, cv2.COLOR_YCrCb2BGR), [cv2.IMWRITE_JPEG_QUALITY, compression_quality]);
		# st, buffger = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_CHROMA_QUALITY, 10, cv2.IMWRITE_JPEG_LUMA_QUALITY, 10, cv2.IMWRITE_JPEG_QUALITY, compression_quality]);
		st, buffger = cv2.imencode(".jpg", resized, [	cv2.IMWRITE_JPEG_CHROMA_QUALITY, 0
														,cv2.IMWRITE_JPEG_PROGRESSIVE, 0
														,cv2.IMWRITE_JPEG_QUALITY, compression_quality 
														,cv2.IMWRITE_JPEG_OPTIMIZE, 1]); # cv2.IMWRITE_FOURCC('I','4','2','0'),
		y.append(len(buffger));
		x.append(i);
		# f_sampling.write('[%d, %d, "%s", %d],\n' %(i, file_size, ntpath.basename(fname), compression_quality));

	plt.xlabel('image width [pixel]');
	plt.ylabel('file size [bytes]');
	plt.plot(x, y);

	# quadratic, linear, intercept = np.polyfit(x, y, 2);
	# metadata = ImageMetadata(fname);
	# metadata.read();
	# key = 'Iptc.Application2.Keywords';
	# metadata[key] = [str(quadratic), str(linear), str(intercept)];
	# metadata.write();


def get_image_size(url):
	data = requests.get(url).content
	#im = Image.open(BytesIO(data))
	out = BytesIO(data)
	out.seek(0, os.SEEK_END)
	return out.tell() 

url_arr = []

url = "https://www.enterprise.ca/content/dam/ecom/general/Homepage/inspiration-banff-ca.jpg.wrend.1280.720.jpeg"
url_arr.append(url)
url = "https://upload.wikimedia.org/wikipedia/commons/1/1f/Nofretete_Neues_Museum.jpg"
url_arr.append(url)
url = "https://www.w3.org/MarkUp/Test/xhtml-print/20050519/tests/jpeg420exif.jpg"
url_arr.append(url)
url = "http://freeforumsigs.com/ffs_gallery/albums/batch/shadows%20animal%20wallpapers/animals_(6).jpg" #hang in there
url_arr.append(url)
url = "https://upload.wikimedia.org/wikipedia/commons/2/2a/Junonia_lemonias_DSF_upper_by_Kadavoor.JPG" #butterfly
url_arr.append(url)
url = "https://upload.wikimedia.org/wikipedia/commons/c/c9/Moon.jpg"
url_arr.append(url)
url = "https://upload.wikimedia.org/wikipedia/commons/4/4b/Everest_kalapatthar_crop.jpg"
url_arr.append(url)
url = "http://www.ustrust.com/publish/content/image/jpeg/GWMOL/panel4-jpg.jpeg"
url_arr.append(url)
url = "https://pbs.twimg.com/media/Bm54nBCCYAACwBi.jpg:large"
url_arr.append(url)
url = "http://www.bigfoto.com/stones-background.jpg"
url_arr.append(url)

url2 = "http://localhost:8888/unsafe/300x0/filters:brightness(10):contrast(30)/https://upload.wikimedia.org/wikipedia/commons/1/1f/Nofretete_Neues_Museum.jpg"
url_thumbor = "http://localhost:8888/unsafe/"

compression_quality = 95

count = 1
amount_to_show = 9

for url in url_arr:
	if count < amount_to_show:
		x = []
		y = []

		data = requests.get(url).content
		img = Image.open(BytesIO(data))
		width = img.size[0]
		step = int(width/4);

		print "width" + str(width)

		## RAW
		time_start = time.time();
		for c in xrange(step,width,step):
			url_res = url_thumbor + str(c) + "x0/filters:quality("+ str(compression_quality) + ")/" + url
			y.append(get_image_size(url_res))
			x.append(c)

		print "raw: " + str(time.time() - time_start)
		plt.plot(x,y)

		## Pillow
		time_start = time.time();
		writeImageMemory(img, step)
		print "pil: " + str(time.time() - time_start)

		## OpenCV
		time_start = time.time();
		writeImageCVMemory(img, step)
		print "opencv: " + str(time.time() - time_start)

		index = 200 + 4*10 + count
		#index = 100 + 1*10 + 1
		plt.subplot(index).set_title(url.rsplit('/', 1)[-1])
		#img.show()
	count += 1

plt.show()






