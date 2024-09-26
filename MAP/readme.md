In the domain of image processing, Maximum A Posteriori (MAP) estimation is a widely used
technique for both image restoration and image segmentation[1]. MAP involves a statistical
method of determining the most probable estimate of the original image by considering both
the observed data and any prior knowledge about the image. Maximum A Posteriori (MAP)
is a Bayesian-based approach to estimating a distribution and model parameters that best
explain an observed data. Optimizing posterior is equivalent to the product of likelihood and
the prior[2]. When MAP is used for the image restoration the original image must have been
degraded by noise, blur effect or other factors. In MAP, the expected feature of the original
image is first determined by the prior probability and then finally those expected features
are added with the degraded image. The features that may represent original image could be
smoothness, or sparseness of the image. MAP could be used as a powerful method to produce
high quality results for noisy or incomplete images when the prior information about that
particular image is accurate and precise.
In this project, we will go through the details of the Maximum A Posteriori (MAP) estimation
for image restoration and implement it in a python programming language. First, we will
discuss about the technical aspects of Maximum A Posteriori (MAP) estimation and then
focus on the Maximum A Posteriori (MAP) estimation for image restoration in section 2.
Section 3 will explain about the experimental details. Section 4 will present the results
obtained from the python implementation of MAP for various images with varying
parameters. Moreover, additional tests are included in subsection 4.5-4.6 that explore the
python implementation of the MAP using extra images. Finally, the project will be concluded
in section 5.
