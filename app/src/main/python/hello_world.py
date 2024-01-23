import tensorflow as tf
#from tensorflow.keras.applications import ResNet101

def generate_fibonacci(msg, num):
    print(msg)
    print(tf.__version__)
    # Generate the first n Fibonacci numbers
    fib_sequence = [0, 1]

    # Load the ResNet101 model pre-trained on ImageNet
    model = tf.keras.applications.ResNet101V2()

    while len(fib_sequence) < num:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])

    print(fib_sequence)
    return msg
