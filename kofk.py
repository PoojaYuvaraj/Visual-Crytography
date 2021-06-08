from __future__ import print_function
import itertools
import numpy as np
import random

def makeW (k):
   W = []
   i = 1;
   while (i <= k):
      W.append(i)
      i += 1
   return W

def makePiSigma (W):
   pi = []
   sigma = []
   for i in range(0, len(W)+1):
      listing = [list(subset) for subset in itertools.combinations(W, i)]
      if (len(listing)%2 == 0):
         pi.extend(listing)
      else:
         sigma.extend(listing)
   #pi[0] =[0]
   return([pi, sigma])

def search (target, search_space):
   if target in search_space:
      return True
   else:
      return False

def makeS (W, ps):
   k = len(W)
   col = pow(2,(k-1))
   s = [[2 for x in range(col)] for y in range(k)]
   for i in range(k):
       for j in range(k):
           if(search(W[i],ps[j])):
               s[i][j] = 0
           else:
               s[i][j] = 1
   return s


def permuteMatrix (matrix):
   matrix = np.array(matrix)
   cols = len(matrix[0])
   for i in range(0,cols):
      rand1 = random.randint(0,cols-1)
      rand2 = random.randint(0,cols-1)
      matrix[:,[rand1,rand2]] = matrix[:,[rand2,rand1]]
   return matrix


def koutofk (k, Matrix):
   W = makeW(k)
   fullset = makePiSigma(W)
   pi = fullset[0]
   sigma = fullset[1]
  
   s0 = makeS(W, pi)
  
   s1 = makeS(W, sigma)

   shares = [object] * k
   for i in range(0, k):
      shares[i] = open("share" + str(i), "w")


   for line in Matrix:
      for pixel in line:

         if pixel == 1:
            out = permuteMatrix(s0) # white pixel
         else:
            out = permuteMatrix(s1) 
         for i in range(0, k):
            for subpixel in out[i]:
               shares[i].write(str(subpixel))
 
      for i in range(0, k):
         shares[i].write("\n")
   for i in range(0, k):
      shares[i].close()

   return 0

def koutofk_to3D_Matrix(k, Matrix):
   if  k % 2 == 0:
      print("Invalid k:",k)
      return [[]]
   W = makeW(k)
   fullset = makePiSigma(W)
   pi = fullset[0]
   sigma = fullset[1]
   s0 = makeS(W, pi)
   s1 = makeS(W, sigma)

   side_len = 2 << ((k-1)//2) - 1  
   #print(side_len)
   pixels = 0
   for pixel in Matrix[0]:
      pixels += 1
   matrix_width = side_len * pixels 
   lines = 0
   for line in Matrix:
      lines += 1
   matrix_depth = side_len * lines
   outMatrix = np.zeros((k, matrix_depth, matrix_width), dtype=np.uint8)
   doffset = 0
   for line in Matrix:
      woffset = 0
      for pixel in line:
       
         if pixel == 1:
            out = permuteMatrix(s0)
         else:
            out = permuteMatrix(s1)
         for i in range(0, k):
            pos = 0
            for depth in range(doffset, doffset + side_len):
               for width in range(woffset, woffset + side_len):
                 
                  outMatrix[i][depth][width] = out[i][pos]
                  pos += 1
         woffset += side_len
      doffset += side_len
   return outMatrix

def toImage(k):

   share = open("share0", "r")
   num_lines = sum(1 for line in share)
   share.close()
   shares = [object] * k
   for i in range(0, k):
      shares[i] = open("share" + str(i), "r")
   length = 2 << (k-2) 
   num_pixels = len(shares[0].readline())/length
   shares[0].seek(0,0)
   Matrix = np.zeros((num_lines, num_pixels), dtype=np.uint8)
   for i in range(0, num_lines):
      lines = [object] * k
      for x in range(0, k):
         lines[x] = shares[x].readline()
         lines[x] = lines[x][:-1] 

      beg = 0 

      while beg < len(lines[0]):
         white=False
         for x in range(beg, beg + length):
            
            w = True
            for line in lines:
               if line[x] != "1":
                  w = False
            if w:
               white=True

         if(white):
            Matrix[i][beg/length] = 1
           
         else:
            Matrix[i][beg/length] = 0
            
         beg += length
 
   return Matrix

def stack_images(Images):
   #set up the output Matrix which will be the dimensions of the original matrix
   num_pixels = sum(1 for subpixel in Images[0][0])
   num_lines = sum(1 for subpixel in Images[0])
   outMatrix = np.zeros((num_lines, num_pixels), dtype=np.uint8)
   for line in range(0, num_lines):
      for pixel in range(0, num_pixels):
         white = True
         for image in Images:
            if image[line][pixel] == 0:
               white = False
         if white:
            outMatrix[line][pixel] = 1
         else:
            outMatrix[line][pixel] = 0
   return outMatrix

def toImage_fr3D(k, Matrix):
   subpixels = 2 << ((k-1) //2) - 1
   num_pixels = sum(1 for subpixel in Matrix[0][0])//subpixels
   num_lines = sum(1 for subpixel in Matrix[0]) //subpixels
   outMatrix = np.zeros((num_lines, num_pixels), dtype=np.uint8)
   for i in range(num_lines):
      for j in range(num_pixels):
         white = False
         for depth in range(i*subpixels, (i+1)*subpixels):
            for width in range(j*subpixels, (j+1)*subpixels):
               w = True
               for share in Matrix:
                  if share[depth][width] == 0:
                     w = False
               if w:
                  white = True
         if white:
            outMatrix[i][j] = 1
         else:
            outMatrix[i][j] = 0
   return outMatrix
