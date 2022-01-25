import tensorflow as tf

# new test functions
print(f"\ngpu avail={tf.test.is_gpu_available()}\n")
print(f"\ngpu name={tf.test.gpu_device_name()}\n")

with tf.Session() as sess:
  devices = sess.list_devices()
  
  # list all devices
  print("\nlist devices:")
  for d in devices:
      print(f"\t{d}")

print()

# run an op on gpu
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))

