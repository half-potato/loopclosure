# Ripped from tensorflow docs
def simple_shuffle_batch(source, capacity, batch_size=10):
  # Create a random shuffle queue.
  queue = tf.RandomShuffleQueue(capacity=capacity,
                                min_after_dequeue=int(0.9*capacity),
                                shapes=source.shape, dtypes=source.dtype)

  # Create an op to enqueue one item.
  enqueue = queue.enqueue(source)

  # Create a queue runner that, when started, will launch 4 threads applying
  # that enqueue op.
  num_threads = 4
  qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

  # Register the queue runner so it can be found and started by
  # `tf.train.start_queue_runners` later (the threads are not launched yet).
  tf.train.add_queue_runner(qr)

  # Create an op to dequeue a batch
  return queue.dequeue_many(batch_size)

def image_loading_tensor():
  return input_strings, queue
