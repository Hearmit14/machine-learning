%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
 
img_eg = mpimg.imread("/Users/hejinyang/Desktop/37.jpg")
print(img_eg.shape)

img_temp = img_eg.reshape(408, 646 * 3)
U,Sigma,VT = np.linalg.svd(img_temp)

sval_nums = 60
img_restruct1 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct2 = img_restruct1.reshape(408,646,3)

sval_nums = 120
img_restruct3 = (U[:,0:sval_nums]).dot(np.diag(Sigma[0:sval_nums])).dot(VT[0:sval_nums,:])
img_restruct4 = img_restruct2.reshape(408,646,3)

fig, ax = plt.subplots(1,3,figsize = (24,32))
 
ax[0].imshow(img_eg)
ax[0].set(title = "src")
ax[1].imshow(img_restruct2.astype(np.uint8))
ax[1].set(title = "nums of sigma = 60")
ax[2].imshow(img_restruct4.astype(np.uint8))
ax[2].set(title = "nums of sigma = 120")

plt.show()

# list = [[[i for i in range(2)] for i in range(3)] for i in range(4)]
# array=np.array(list)
# print(array.shape)
# array1 = array.reshape(4, 3 * 2)

# array1[:,0:1]
# array1[0:1,:]