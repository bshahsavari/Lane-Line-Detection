#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

#%% 
###Camera Calibration
    #The camera is calibrated using a set of "object points" and "image points" that 
    #are extracted from calibration images in "./camera_cal/" directory. The image points 
    #are extracted from the calibration images by an OpenCV function called `findChessboardCorners()' 
    #and visualized with `drawChessboardCorners()` again available in OpenCV. 
    #The object points are basically set to be a uniform grid of 9x6 based on the fact 
    #that the chessboard had the same number of grid points.
    #
    #Once the image points and corresponding object points are found, the  camera 
    #calibration parameters and distortion matrix are determined by `ret, mtx, dist, 
    #rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)` 
    #where `objpoints` and `imgpoints` are the arrays containing the object points and 
    #image points respectively.
    #
    #The images can then be *undistorted*  by using the distortion matrix and OpenCV 
    #function `dst = cv2.undistort(img, mtx, dist, None, mtx)`. Examples of calibration 
    #and undistorted images are shown below.
                                                                                                             
nx = 9
ny = 6
# prepare object points
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

# Calibrate camera using the calculated object points and iamge points        
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
    img2 = cv2.drawChessboardCorners(np.copy(img), (9,6), corners, ret)
    
    plt.subplot(1,2,1)
    plt.imshow(img2)
    plt.title('Original Image')
    plt.subplot(1,2,2)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.title('Undistorted Image')
    # Uncomment to save the images in ./output_images/
    #plt.savefig('./output_images/'+fname.split('/')[-1])
    plt.show()


# The same transformation is applied on the video to undistot the frames. 
#    The following figures show two examples of original and undistorted images. 
#    Note that most of the objects are farther than the chessboard the undistortion 
#    effect is less apparent in this images compared to the calibration ones.

images = glob.glob('./test_images/*.jpg')
for fname in images:
    #reading in an image
    img = mpimg.imread(fname)
    
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Original Image')
    
    plt.subplot(1,2,2)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.title('Undistorted Image')
    # Uncomment to save the images in ./output_images/
    # plt.savefig('./output_images/'+fname.split('/')[-1])
    plt.show()


#%%

def s_threshold(img,thresh=(0,255)):
    s_thresh_min, s_thresh_max = thresh
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    # Threshold color channel
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    return s_binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    thresh_min, thresh_max = thresh
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def plot_windows(img, out_img,  leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    left_curverad, right_curverad, offset, left_fit_cr, right_fit_cr = calculate_curvature_offset(binary_warped, leftx, lefty, rightx, righty)
    left_fit_str = 'Left line: %.4f Y^2 + %.3f Y + %.1f' %(left_fit_cr[0], left_fit_cr[1], left_fit_cr[2])
    right_fit_str = 'Right line: %.4f Y^2 + %.3f Y + %.1f' %(right_fit_cr[0], right_fit_cr[1], right_fit_cr[2])
    # Now our radius of curvature is in meters
    ax = plt.gca()
    strcurve = 'Avg. curvature: %.1f m ' %(0.5*(left_curverad+right_curverad))
    stroffset = 'Center of car relative to lane center: %.1f m ' %(offset)
    ax.text(img.shape[1]/3, img.shape[0]/5,strcurve, color='white')
    ax.text(img.shape[1]/3, img.shape[0]/5+70,stroffset, color='white')
    ax.text(img.shape[1]/3, img.shape[0]/5+140,left_fit_str, color='white')
    ax.text(img.shape[1]/3, img.shape[0]/5+210,right_fit_str, color='white')

def plot_lines(img, out_img,  leftx, lefty, rightx, righty):
        # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def calculate_curvature_offset(img, leftx, lefty, rightx, righty):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
   
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    y_eval_cr = y_eval*ym_per_pix
    left_fitx = left_fit_cr[0]*y_eval_cr**2 + left_fit_cr[1]*y_eval_cr + left_fit_cr[2]
    right_fitx = right_fit_cr[0]*y_eval_cr**2 + right_fit_cr[1]*y_eval_cr + right_fit_cr[2]
    offset = img.shape[1]/2*xm_per_pix - 0.5*(left_fitx+right_fitx)
    return left_curverad, right_curverad, offset, left_fit_cr, right_fit_cr

def write_stats(img, out_img,  leftx, lefty, rightx, righty):
    left_curverad, right_curverad, offset, left_fit_cr, right_fit_cr = calculate_curvature_offset(binary_warped, leftx, lefty, rightx, righty)
    left_fit_str = 'Left line: %.4f Y^2 + %.3f Y + %.1f' %(left_fit_cr[0], left_fit_cr[1], left_fit_cr[2])
    right_fit_str = 'Right line: %.4f Y^2 + %.3f Y + %.1f' %(right_fit_cr[0], right_fit_cr[1], right_fit_cr[2])
    # Now our radius of curvature is in meters
    strcurve = 'Avg. curvature: %.1f m ' %(0.5*(left_curverad+right_curverad))
    stroffset = 'Center of car relative to lane center: %.1f m ' %(offset)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, stroffset, (int(img.shape[1]/3), int(img.shape[0]/5)), font, 0.8, (0,0,255),2,cv2.LINE_AA)
    cv2.putText(img, strcurve, (int(img.shape[1]/3), int(img.shape[0]/5)+40), font, .8, (0,0,255),2,cv2.LINE_AA)
    cv2.putText(img, left_fit_str, (int(img.shape[1]/3), int(img.shape[0]/5)+80), font, .8, (0,0,255),2,cv2.LINE_AA)
    cv2.putText(img, right_fit_str, (int(img.shape[1]/3), int(img.shape[0]/5)+120), font, .8, (0,0,255),2,cv2.LINE_AA)
    return img

leftx_base = np.nan
rightx_base = np.nan

def find_lines(binary_warped):
    global leftx_base
    global rightx_base
    alpha2 = 0.05
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    tl = np.argmax(histogram[:midpoint])
    tr = np.argmax(histogram[midpoint:]) + midpoint
                           
    if np.isnan(leftx_base):
        leftx_base = tl
        rightx_base = tr
    else:
        leftx_base = (1.-alpha2)*leftx_base + alpha2 * tl
        rightx_base = (1.-alpha2)*rightx_base + alpha2 * tr
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = int(leftx_base)
    rightx_current = int(rightx_base)
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    return out_img, leftx, lefty, rightx, righty

left_fit = []
right_fit = []

def plot_poly(warped, undist, leftx, lefty, rightx, righty):
    global left_fit
    global right_fit
    alpha = 0.2
    # Fit a second order polynomial to each
    tl = np.polyfit(lefty, leftx, 2)
    tr = np.polyfit(righty, rightx, 2)
    if len(left_fit) == 0:
        left_fit = tl
        right_fit = tr
    else:
        left_fit = (1.-alpha)*left_fit + alpha * tl
        right_fit = (1.-alpha)*right_fit + alpha * tr
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def combine_thresholds(img):
    # Apply each of the thresholding functions
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(50, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(100, 190))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(0, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0, np.pi/2))
    s_binary = s_threshold(img,(170,255))
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary, color_binary
#%%
def perspective_transformation(img):
    imshape = img.shape
    x1 = 200
    oh = 20
    h = 450
    src = np.array([(x1,imshape[0]-oh),(620, h), (730, h), (imshape[1],imshape[0]-oh)], dtype=np.float32)
    dst = np.array([(x1,imshape[0]-oh),(x1, 0), (imshape[1], 0), (imshape[1],imshape[0]-oh)], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (img.shape[1], img.shape[0])
    return M, Minv, img_size, src, dst


#% Example of the end-to-end process
ksize = 3

# 0- Read an image
img = mpimg.imread('./test_images/test3.jpg')

# 1- Undistort the image using camera calibration matrix
undist = cv2.undistort(img, mtx, dist, None, mtx)

# 2- Threshold image
combined_binary, color_binary = combine_thresholds(undist)

# 3- Perspective transformation
M, Minv, img_size, src, dst = perspective_transformation(combined_binary)
binary_warped = cv2.warpPerspective(combined_binary, M, img_size)

# Plotting thresholded images
plt.subplot(1,2,1)
plt.imshow(color_binary*255)
plt.title('Stacked thresholds')
plt.subplot(1,2,2)
plt.imshow(combined_binary, cmap='gray')
plt.title('Combined S channel and Grad. thresholds')
# plt.savefig('./output_images/thresholded_2.jpg')
plt.show()


# Plot the source and destination points of perspective transformation
plt.imshow(img,cmap='gray')
plt.hold(1)
plt.plot(src[:,0],src[:,1])
plt.plot(dst[:,0],dst[:,1],'r')
#plt.savefig('./output_images/src_dst_2.jpg')
plt.show()

# Plot the bird's eye view image
img_warped = cv2.warpPerspective(img, M, img_size)
plt.imshow(img_warped,cmap='gray')
#plt.savefig('./output_images/warped_2.jpg')
plt.show()


#%
# Set the width of the windows +/- margin
margin = 100




out_img, leftx, lefty, rightx, righty = find_lines(binary_warped)
plot_windows(binary_warped, out_img, leftx, lefty, rightx, righty)
#plt.savefig('./output_images/windows_2.jpg')
plt.show()

plot_lines(binary_warped, out_img, leftx, lefty, rightx, righty)
plt.show()

result = plot_poly(binary_warped, undist, leftx, lefty, rightx, righty)
plt.imshow(result)
#plt.savefig('./output_images/final_2.jpg')




result = write_stats(result, out_img,  leftx, lefty, rightx, righty)
plt.imshow(result)

#%%
def process_image(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    combined_binary, color_binary = combine_thresholds(undist)
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size)
    out_img, leftx, lefty, rightx, righty = find_lines(binary_warped)
    result = plot_poly(binary_warped, undist, leftx, lefty, rightx, righty)
    result = write_stats(result, out_img,  leftx, lefty, rightx, righty)
    return result

file_output = './output_images/project_output2.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(file_output, audio=False)

#%%
file_output = './output_images/project_output.gif'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_gif(file_output,fps=5,opt='nq')






