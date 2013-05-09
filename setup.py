from distutils.core import setup

setup(name='zmqnumpy',
      version='0.1',
      author="Marco Bartolini",
      author_email = "marco.bartolini@gmail.com",
      url = "http://www.med.ira.inaf.it/~mbartolini/zmqnumpy/",
      download_url = "",
      license = "mit",
      py_modules = ['zmqnumpy'],
      requires = ["numpy", "pyzmq"],
      classifiers = [
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          ],
      description="numpy array over zmq sockets",
      long_description = """
Zmqnumpy module implements a series of functions used to exchange
numpy ndarrays between U{zeromq<http://www.zeromq.org>} sockets.
Serializtion of numpy arrays happens using the numpy.ndarray.tostring method
which preserves portability to standard C binary format, 
enabling data exchange with different programming languages.
A very simple protocol is defined in order to exchange array data, the
multipart messages will be composed of:

1. identifier string name
2. the numpy array element type (dtype) in its string representation
3. numpy array shape encoded as a binary numpy.int32 array 
4. the array data encoded as string using numpy.ndarray.tostring()

This protocol guarantees that numpy array can be carried around and
recostructed uniquely without errors on both ends of a connected pair enabling
an efficient interchange of data between processes and nodes.

@author: Marco Bartolini
@contact: marco.bartolini@gmail.com
@version: 0.1
      """,
     )
