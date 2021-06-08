
import numpy as np

from imageio import imwrite

from PIL import Image
import kofk

def image_to_bits(image):
	print ("Image info:")
	print ("""Number of bits: %d,  Image Size: %s,  Image format: %s""" %(image.bits, image.size, image.format))
	image.show()
	image = image.convert("L")
	toBW(image)
	return np.array(image.convert("1").getdata())

def toBW(image):
	depth, width = image.size
	pixels = image.load()
	for row in range(depth):
		for pixel in range(width):
			if pixels[row, pixel] >= 127:
				pixels[row, pixel] = 255
			else:
				pixels[row, pixel] = 0

def print_image(image):
	image.show()


def print_bit_array(image, data):
	count = 0
	for bit in data:
		count += 1

		if count % image.size[1] == 0:
			print(bit/255)

		else:
			print(bit/255)


def flip_bits(bits):
	for x in range(len(bits)):
		if bits[x] == 255:
			bits[x] = 0
		else:
			bits[x] = 255


def paste_images(background, foreground):
	return Image.alpha_composite(background, foreground).save("stacked-img.png")

def from_2D_to_img(Matrix):
	for row in range(len(Matrix)):
		for pixel in range(len(Matrix[row])):
			if Matrix[row][pixel] == 1:
				Matrix[row][pixel] = 255

def make_2D_array(data,ratio):
	print(ratio[0],ratio[1])
	Matrix = np.zeros((ratio[1], ratio[0]), dtype=np.uint8)
	x,y = 0,0
	for datum in data:
		Matrix[y][x] = datum + 1
		x = (x+1) % ratio[0]
		if x == 0:
			y = (y+1) % ratio[1]
	return Matrix

if __name__ == '__main__':
	
	k = 3
	inp = Image.open('E:/Pooja/visual cryptography/VisualCryptography-master/Rose.jpg')										# Open image
	out = image_to_bits(inp)										
	flip_bits(out)
	Matrix = make_2D_array(out,inp.size)
	shares = kofk.koutofk_to3D_Matrix(k,Matrix)
	for i in range(k):
		from_2D_to_img(shares[i])
		imwrite("share"+str(i)+".jpg", shares[i])
	outMatrix = kofk.toImage_fr3D(k, shares)
	from_2D_to_img(outMatrix)
	imwrite('result.jpg', outMatrix)
	from_2D_to_img(Matrix)
	imwrite('beforeShare.jpg', Matrix)

	outMatrix = kofk.stack_images(shares)
	from_2D_to_img(outMatrix)
	imwrite('stacked.jpg', outMatrix)