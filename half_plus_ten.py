import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter


def Export():
  export_path = "model/half_plus_ten"
  with tf.Session() as sess:
    # Make model parameters a&b variables instead of constants to
    # exercise the variable reloading mechanisms.
    a = tf.Variable(0.5)
    b = tf.Variable(10.0)

    # Calculate, y = a*x + b
    # here we use a placeholder 'x' which is fed at inference time.
    x = tf.placeholder(tf.float32)
    y = tf.add(tf.multiply(a, x), b)

    # Run an export.
    tf.global_variables_initializer().run()
    export = exporter.Exporter(tf.train.Saver())
    export.init(named_graph_signatures={
        "inputs": exporter.generic_signature({"x": x}),
        "outputs": exporter.generic_signature({"y": y}),
        "regress": exporter.regression_signature(x, y)
    })
    export.export(export_path, tf.constant(123), sess)


def main(_):
  Export()

if __name__ == "__main__":
	tf.app.run()