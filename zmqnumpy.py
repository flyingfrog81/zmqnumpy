# Copyright (c) 2012 Marco Bartolini, marco.bartolini@gmail.com
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#

"""
This module implements a series of functions used to exchange
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
"""
import numpy
import zmq
import functools
import uuid

def array_to_msg(nparray):
    """
    Convert a numpy ndarray to its multipart zeromq message representation.
    The return list is composed of:
      0. The string representation of the array element type, i.e. 'float32'
      1. The binary string representation of the shape of the array converted to a numpy array with dtype int32
      2. The binary string representation of the array
    These informations together can be used from the receiver code to recreate
    uniquely the original array.
    @param nparray: A numpy ndarray
    @type nparray: numpy.ndarray
    @rtype: list
    @return: [dtype, shape, array]
    """
    _shape = numpy.array(nparray.shape, dtype=numpy.int32).tostring()
    return [nparray.dtype.name, _shape, nparray.tostring()]

def msg_to_info(msg):
    _shape = numpy.fromstring(msg[1], dtype=numpy.int32)
    return [msg[0], _shape, msg[2]]

def msg_to_array(msg):
    """
    Parse a list argument as returned by L{array_to_msg} function of this
    module, and returns the numpy array contained in the message body.
    @param msg: a list as returned by L{array_to_msg} function
    @rtype: numpy.ndarray
    @return: The numpy array contained in the message
    """
    [_dtype, _shape, _bin_msg] = msg_to_info(msg)
    return numpy.fromstring(_bin_msg, dtype=_dtype).reshape(tuple(_shape))

def numpy_array_sender(name, endpoint, socket_type=zmq.PUSH):
    """
    Decorator Factory
    The decorated function will have to return a numpy array, while the
    decorator will create a zmq socket of the specified socket type connected
    to the specified endpoint.
    Each time the function is called the numpy array will be sent over the
    instantiated transport after being converted to a multipart message using
    L{array_to_msg} function. The multipart message is prepended with a UUID
    and the given name as the first two elements.
    #TODO: Would it be good to add the possibility of transimitting arbitrary
    metadata? --- Marco Bartolini 27/04/2012

    Usage example::

        import zmq
        import zmqnumpy
        import numpy

        @zmqnumpy.numpy_array_sender(\"mysender\", \"tcp://127.0.0.1:8765\")
        def random_array_generator(min, max, width):
            return numpy.random.randint(min, max, width)

    @type name: string
    @param name: the label of the data stream
    @type endpoint: string
    @param endpoint: a zmq endpoint made as \"protocol://host:port\"
    @param socket_type: a zmq socket type such as zmq.PUSH or zmq.PUB

    """
    _context = zmq.Context.instance()
    _socket = _context.socket(socket_type)
    _socket.connect(endpoint)
    _uuid = uuid.uuid4().bytes
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            _data = fn(*args, **kwargs)
            _socket.send_multipart([_uuid, name] + array_to_msg(_data))
        return wrapped
    return wrapper

