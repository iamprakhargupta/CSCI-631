Assignment 01 - Getting started / introductions
===============================================

## Problem 1: who are you? (5pts)

1. Upload a picture of yourself to your MyCourses profile if you haven't already

2. Go to the "introductions" discussion board on MyCourses and introduce yourself to your classmates. To do this, click on the "New > Topic" button at the top of the page. In the post, say a bit about who you are, where you're from, what you like to do for fun, and what you hope to learn / what you hope to be able to do by the end of this course. I created an example introduction post to get things started.

## Problem 2: Python environment setup (5pts)

1. Choose an IDE that you would like to use for the course. The most popular choices are Visual Studio Code or PyCharm, but the choice is ultimately up to you. Big selling points of VSCode and PyCharm are that they both fairly seamlessly support Github Copilot, and they both fairly seamlessly support running and debugging code remotely over SSH.

2. Use your IDE to install Python 3.10 or later (version 3.10 introduced some nice syntax for type hints that is really nice but isn't backwards compatible with earlier versions)

3. Hello, world!

    Create a new project in your IDE for assignment 01 and copy `requirements.txt` and `hello.py` into it. Figure out how to install the packages listed in `requirements.txt` in your project. For PyCharm users, this is usually as simple as copying the requirements file into the project and then clicking "ok" on the dialog box that pops up asking if you want to install the packages.

    To see if it worked, run the `hello.py` file. This will do 3 things:

    1. it will turn on your webcam and take a picture of you
    2. it will calculate a histogram of reds, greens, and blues in the image and display it using matplotlib
    3. it will display an image back to you and save it out as `hello.jpg`. You can then press any key to exit the program.

    Note that I'm a little old-fashioned in that I prefer using python virtual environments rather than Conda environments. You are welcome to use conda or miniconda for this course if you prefer.

    __Include a copy of the hello.jpg when you upload your assignment__

## Problem 3: Remote SSH setup (5pts)

1. By enrolling in this course, you should now have access to three CS department servers:

        granger.cs.rit.edu
        weasley.cs.rit.edu
        lovegood.cs.rit.edu

    and you should be able to log in to them using SSH and your RIT username. __These servers are shared with faculty and with students in other classes -- please be respectful of others' time and resource needs__. A guide can be found [here](https://wiki.cs.rit.edu/index.php/Using_gpu_nodes).

    We're going to repeat some of the setup from Problem 2, but this time getting things running on the servers. First, pick one of the above three servers at random. Either using your IDE or from the command line, create another Python environment on the server (for PyCharm or VSCode, you can configure a new python intepreter that works over SSH), and install the requirements again. Then, run the `gpu_test.py` script and paste the output below:

        Output of gpu_test.py:
        ...
        ...
        ...
        ...
        ...
        ...

2. Prove that you know how to locate and copy files on a remote server. There is a file called `treasure.jpg` in the `/local/sandbox/csci631` directory on the `lovegood` server. Copy it to your machine using `scp` or `rsync` and include it in this directory when you upload your assignment.

## Collaboration and Generative AI disclosure

Did you collaborate with anyone? Did you use any Generative AI tools? Briefly explain what you did here.

## Submitting

Your submission should consist of a single zip file named like `firstname_lastname_assignment1.zip`. The zip file should contain all the contents of this directory, including this file (which you should have written some answers into above) and two images: `hello.jpg` and `treasure.jpg`. Upload the zipped directory to MyCourses.