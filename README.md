# User Instructions 

### Instructions for running the tutorials at Cork AI Meetup2

#### 1: AMAZON WEB SERVICES (AWS):  
**Sign in to your AWS account**
 - From https://aws.amazon.com/ choose "Sign in to the Console"

**Launching AWS virtual machine:**
 - Go to "Services" and under the "compute" heading, choose "EC2"
 - Set "region" in top-right corner to be Ireland
 - Click on "Launch Instance"
 - Scroll down and select "Deep Learning AMI (Ubuntu)" AMI ID: ami-5bf34b22
 - Scroll down and select "GPU compute ... p2.xlarge"
 - Click "Review and Launch"
 - Click "Launch"
 - If you do not have an existing key pair, then select "Create a new key pair".  This will direct you to create and download a .pem file to your disk. Otherwise select an existing key pair. Note that you must have access to the key pair PEM file locally.
 - Click "Launch Instances"

**Connecting to the launched instance:**
 - From Services menu choose EC2
 - From EC2 dashboard->instances
 - You should see your launched instance listed
 - To connect to the instance (using linux, mac or cygwin with openSSH setup) 
   - copy public DNS(ipv4) field
   - open a shell and type ```ssh -i /path/my-key-pair.pem ubuntu@[copied-DNS]```
   (you may need to type ```chmod 400 /path/my-key-pair.pem``` if your key_pair permissions are incorrect) 
(If in doubt, see also http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)
 - To connect to the instance using putty on Windows, please follow directions at http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html

Now you should be logged into the machine and see a command-line prompt $.

#### 2: Review of convolutional neural network from last week
**Folder setup**
 Type the following commands to get setup for running the code:
 - ```mkdir cork_ai```   *(make a new folder to work in)*
 - ```cd cork_ai```         *(switch to the newly created folder)*
 - ```git clone https://github.com/CorkAI/Meetup2.git```  *(this will make a Meetup2 folder with all the code/data we need)*
 - ```cd Meetup2```     *(switch to the Meetup2 folder)*
 - To have a look at the tensorflow code, type ```vim mnist_deep.py```
    - (Type Esc then : then q! and hit enter to exit the file)  

**Launch conda environment**
 Our AWS machine has multiple deep-learning environments installed (conda environments).  We need to launch one so that the libraries we need (e.g. tensorflow) are ready for use:  
 - Type ```source activate tensorflow_p27```

**Executing code**
- ```python mnist_deep.py``` *(and wait, it runs for 20,000 steps)*  
 - Verify the accuracy printed at the end of the file (~99.2%)

#### 3: Pizza and Beer Break  
**Please refuel before the next session :-) **

#### 4: Convnets in Keras
(Slides on keras available at https://slides.com/keelinm/keras/)  
 - Before we run it let's have a look at how quickly this network comes together with keras:
 - Type ```vim keras_mnist_deep.py```
    - (Type Esc then : then q! and hit enter to exit the file)  

**Executing code**
 - Check that it achieves the same result as the raw tensorflow network:
    -  Type ```python keras_mnist_deep.py``` *(and wait, it runs for 20,000 steps)*  
    - Verify the accuracy printed at the end of the file (~99.2%)  

**Additional exercises** 
*Additional Exercise 1:* Run the stored trained convolutional network on the (pre-prepared): handwritten digits provided
The git repository contains some manually created test images in folder extra_test_digits.
Images 1.jpg 2.jpg and 3.jpg are created from photos of handwritten digits (manipulated to be grey-scale, 28x28, white-on-black).  The original photographs are also available to view at 1_photo.jpg etc.
Images 4.jpg 5.jpg and 6.jpg are created digitally using a 28x28 black background and white 'paintbrush'.
Have a look at the images and see how closely they resemble the MNIST data (samples in your output_images folder if you have done previous additional exercises).
Now test your trained convolutional network on these images using the following commands
 - ```pip install pillow``` *(install the python image library, pillow, needed to read the images)*
 - ```python keras_mnist_deep.py --extra_test_imgs 1```
Output files are written in folder output_images with filename extraID_[pred] where ID is the original image name and pred is the digit assigned by the convolutional network. 
 - Use scp to copy the output images to your local machine for inspection:
 	- (linux, mac, cygwin): open a new shell on your local machine and create a fresh empty directory. Then copy the output images to your local system:
		- ```mkdir output_images```
		- ```cd output_images```
		- ```scp -i /path/my-key-pair.pem ubuntu@[copied-DNS]:/home/ubuntu/cork_ai/Meetup2/output_images/* .```
		- View the images using Finder / Explorer or your preferred image viewer.
	- (putty on Windows): Open a command line prompt (cmd)
		- ```pscp -i C:\path\my-key-pair.ppk ubuntu@[copied-DNS]:/home/ubuntu/cork_ai/Meetup2/output_images/* c:\[my_local_directory]```
		- View the images using your preferred image viewer

*Additional Exercise 2:*  Run the convolutional network on the Fashion MNIST data. Note that you will have to re-train using the fashion data, so first delete or rename the saved_model folder which contains the network trained on MNIST digit data.
  - ```mv saved_model/ saved_model_digits/``` *(rename the saved_model folder to saved_model_digits)*
  - ```python keras_mnist_deep.py --data_dir data/fashion --write_samples 1``` *(retrain and test using fashion data)*
The accuracy on the fashion dataset should be significantly lower than on the mnist one since the data is more difficult.  Sample output files are in the output_images folder with the prefix 'fashion_deep' with the true label followed by the predicted label (if different).


#### 5: Ending your AWS session
When you are finished working on AWS you need to stop (or terminate) your AWS instance to discontinue usage charges.
This is **not** achieved by just logging out in the terminal!!

**Stopping/Terminating your instance.**
- From EC2 dashboard->instances 
 - You should see your launched instance listed (and selected with blue checkbox)
 - In the "Actions" drop-down menu choose "Instance State" and either "stop" or "terminate"
   - "stop" will end your session, but keep your instance and data safe for next time you want to use it. The fee for maintaining the data volume only will be around $5.50 per month.
   - "terminate" will end your session and will **not** retain your data or your instance state. There will be no further charge on your account if you choose terminate

