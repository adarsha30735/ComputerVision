#!/usr/bin/env python
# coding: utf-8

# # MAP Implementation

# In[73]:


def restore_image_gradient_descent(image_name, sigma, alpha, prior_mean,prior_variance,learning_rate,num_iterations ):
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
   # Read the image file
    img = Image.open(image_name).convert('L') # 'L' mode for grayscale
    image = np.array(img) / 255.0

   # Add noise to the image
    noisy_image = image + np.random.normal(0, sigma, size=image.shape)

   # Initialize the restored image
    restored_image = noisy_image.copy()

   # Define the gradient descent update function with Gaussian prior
    def gradient_descent_update(image, noisy_image, alpha, sigma, learning_rate, prior_mean, prior_variance):
       # Compute the gradient of the posterior distribution
        posterior = np.exp(-0.5 * ((image - noisy_image) / sigma)**2)
        gradient = (noisy_image - image) / sigma**2 - alpha * (image - prior_mean) / prior_variance
        gradient *= posterior
        image += learning_rate * gradient
        image = np.clip(image, 0, 1)
        return image

    # Perform gradient descent
    for i in range(num_iterations):
        restored_image = gradient_descent_update(restored_image, noisy_image, alpha, sigma, learning_rate, prior_mean, prior_variance)

    # Display the results
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Noisy image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(restored_image, cmap='gray')
    plt.title('Restored image')
    plt.axis('off')

    plt.show()

   # Compute the PSNR of the noisy image
    mse_noisy = np.mean((noisy_image - image)**2)
    psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy))
    print(f"Noisy image PSNR: {psnr_noisy:.2f} dB")

   # Compute the PSNR of the restored image
    mse_restored = np.mean((restored_image - image)**2)
    psnr_restored = 20 * np.log10(1.0 / np.sqrt(mse_restored))
    print(f"Restored image PSNR: {psnr_restored:.2f} dB")

    return noisy_image, restored_image


# ## Experiment 4.1

# In[78]:


noisy_image,restored_image=restore_image_gradient_descent("image.png", 0.1, 0.01, 0,0.01, 0.01,1000)
    
# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(noisy_image*255))
noisy_img.save('Noisy_G_result_1.png')
restored_img = Image.fromarray(np.uint8(restored_image*255))
restored_img.save('Restored_G_result_1.png')


# ## Experiment 4.2

# In[80]:


noisy_image2,restored_image2 = restore_image_gradient_descent("image2.png", 0.09, 0.01, 0,0.01, 0.001,800)

# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(noisy_image2*255))
noisy_img.save('Noisy_G_result_2.png')
restored_img = Image.fromarray(np.uint8(restored_image2*255))
restored_img.save('Restored_G_result_2.png')


# ## Experiment 4.3

# In[87]:


noisy_image3,restored_image3 = restore_image_gradient_descent("moon.jpg", 0.085, 0.01, 0,0.01, 0.01,1200)

# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(noisy_image3*255))
noisy_img.save('Noisy_G_result_3.png')
restored_img = Image.fromarray(np.uint8(restored_image3*255))
restored_img.save('Restored_G_result_3.png')


# ## Experiment 4.4

# In[91]:


noisy_image4,restored_image4 = restore_image_gradient_descent("elk.jpg", 0.12, 0.01, 0,0.01, 0.01,1200)

# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(noisy_image4*255))
noisy_img.save('Noisy_G_result_4.png')
restored_img = Image.fromarray(np.uint8(restored_image4*255))
restored_img.save('Restored_G_result_4.png')


# ## Experiment 4.5

# In[97]:


noisy_image5,restored_image5 = restore_image_gradient_descent("scooter.tiff", 0.07, 0.01, 0,0.01, 0.01,650)

# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(noisy_image5*255))
noisy_img.save('Noisy_G_result_5.png')
restored_img = Image.fromarray(np.uint8(restored_image5*255))
restored_img.save('Restored_G_result_5.png')


# # Experiment 4.6

# In[98]:


noisy_image6,restored_image6 = restore_image_gradient_descent("building.tiff", 0.065, 0.01, 0,0.01, 0.01,1000)

# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(noisy_image6*255))
noisy_img.save('Noisy_G_result_6.png')
restored_img = Image.fromarray(np.uint8(restored_image6*255))
restored_img.save('Restored_G_result_6.png')


# ## 4.7 Effect of using  MFA and different priors

# In[101]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def likelihood(y, x, sigma):
    
    diff = y - x
    return np.exp(-0.5 * np.sum(diff**2) / sigma**2)

def prior(x, beta):
    
    #using piecewise constant
    
    dy, dx = np.gradient(x)
    grad_norm = np.sqrt(dx**2 + dy**2)
    return np.exp(-beta * np.sum(np.abs(grad_norm)))


def energy(y, x, sigma, beta):
    
    return -np.log(likelihood(y, x, sigma)) - np.log(prior(x, beta))

def MFA_opt(y, sigma, beta, temperature, num_iterations):
    
    x = y.copy()
    T = temperature
    for i in range(num_iterations):
        # Update the temperature
        T = temperature * (num_iterations-i) / num_iterations
        # Compute the energy function
        E = energy(y, x, sigma, beta)
        # Update the image using MFA
        dy, dx = np.gradient(x)
        grad_norm = np.sqrt(dx**2 + dy**2)
        prior_grad = -beta * np.sign(grad_norm)
        likelihood_grad = (y - x) / sigma**2
        x += T * (prior_grad + likelihood_grad)
    return x


# In[102]:


# Load the image
img = Image.open('scooter.tiff').convert('L') # 'L' mode for grayscale
image = np.array(img) / 255.0

# Define the noise model
sigma = 0.1
# Add noise to the image
y = image + np.random.normal(0, sigma, size=image.shape)


# Set the parameters
sigma = 11
beta = 0.75
temperature = 90
num_iterations = 900

# Run mean field annealing
x = MFA_opt(y, sigma, beta, temperature, num_iterations)

# Compute the PSNR of the noisy image
mse_noisy = np.mean((y - image)**2)
psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy))
print(f"Noisy image PSNR: {psnr_noisy:.2f} dB")

# Compute the PSNR of the restored image
mse_restored = np.mean((x - image)**2)
psnr_restored = 20 * np.log10(1.0 / np.sqrt(mse_restored))
print(f"Restored image PSNR: {psnr_restored:.2f} dB")

# Save the noisy image and the restored images
noisy_img = Image.fromarray(np.uint8(y*255))
noisy_img.save('Noisy_MFA_result_7.png')
restored_img = Image.fromarray(np.uint8(x*255))
restored_img.save('Restored_MFA_result_7.png')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def likelihood_Probability(y, x, sigma):
   
    diff = y - x
    return np.exp(-0.5 * np.sum(diff**2) / sigma**2)

def image_prior(x, beta):
    
    #use laplacian prior
    
    dy, dx = np.gradient(x)
    laplacian = np.abs(dx) + np.abs(dy)
    return np.exp(-beta * np.sum(laplacian))

def energy_equation(y, x, sigma, beta):
    
    return -np.log(likelihood_Probability(y, x, sigma)) - np.log(image_prior(x, beta) + 1e-10)


def MFA_OPT(y, sigma, beta, temperature, num_iterations):
  
    # Initialize the image
    x = y.copy()
    # Initialize the temperature
    T = temperature
    # Run the algorithm
    for i in range(num_iterations):
        # Update the temperature
        T = temperature * (num_iterations-i) / num_iterations
        # Compute the energy function
        E = energy_equation(y, x, sigma, beta)
        # Update the image using mean field annealing
        dy, dx = np.gradient(x)
        laplacian = np.abs(dx) + np.abs(dy)
        prior_grad = -beta * np.sign(laplacian)
        likelihood_grad = (y - x) / sigma**2
        x += T * (prior_grad + likelihood_grad)
    return x


# In[3]:


# Load  image
img = Image.open('building.tiff').convert('L') # 'L' mode for grayscale
image = np.array(img) / 255.0

# Add noise 
sigma = 0.15
y = image + np.random.normal(0, sigma, size=image.shape)

# parameters
sigma = 17
beta = 0.66
temperature = 70
num_iterations = 850

# Call MFA
x = MFA_OPT(y, sigma, beta, temperature, num_iterations)

# PSNR of the noisy image
mse_noisy = np.mean((y - image)**2)
psnr_noisy = 20 * np.log10(1.0 / np.sqrt(mse_noisy))
print(f"Noisy image PSNR: {psnr_noisy:.2f} dB")

#  PSNR of the restored image
mse_restored = np.mean((x - image)**2)
psnr_restored = 20 * np.log10(1.0 / np.sqrt(mse_restored))
print(f"Restored image PSNR: {psnr_restored:.2f} dB")


# In[ ]:




