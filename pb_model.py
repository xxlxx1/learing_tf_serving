import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

class model():
    def __init__(self):
        self.a = tf.placeholder(tf.float32, [None])
        self.w = tf.Variable(tf.constant(2.0, shape=[1]), name="w")
        b = tf.Variable(tf.constant(0.5, shape=[1]), name="b")
        self.y = self.a * self.w + b

#模型保存为ckpt
def save_model():
    graph1 = tf.Graph()
    with graph1.as_default():
        m = model()
        sess1 = tf.Session()
        sess1.run(tf.global_variables_initializer())
        update = tf.assign(m.w, [10])
        sess1.run(update)
        predict_y = sess1.run(m.y,feed_dict={m.a:[3.0]})
        print(predict_y)

        saver = tf.train.Saver()
        saver.save(sess1,"model_pb/model.ckpt")
        sess1.close()

#加载ckpt模型
def load_model():
    graph2 = tf.Graph()
    with graph2.as_default():
        m = model()
        session = tf.Session()
        saver = tf.train.Saver()
        saver.restore(session, "model_pb/model.ckpt")
        predict_y = session.run(m.y, feed_dict={m.a: [3.0]})
        print(predict_y)
        return session,m

#保存为pb模型
def export_model(session,m):

   #只需要修改这一段，定义输入输出，其他保持默认即可
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.a)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)

    export_path = "pb_model/1"
    print("Export the model to {}".format(export_path))

    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))

#加载pb模型
def load_pb():
    session = tf.Session(graph=tf.Graph())
    model_file_path = "pb_model/1"
    meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_file_path)

    model_graph_signature = list(meta_graph.signature_def.items())[0][1]
    output_tensor_names = []
    output_op_names = []
    for output_item in model_graph_signature.outputs.items():
        output_op_name = output_item[0]
        output_op_names.append(output_op_name)
        output_tensor_name = output_item[1].name
        output_tensor_names.append(output_tensor_name)
    print("load model finish!")
    sentences = {}
    # 测试pb模型
    for test_x in [[1],[2],[3],[4],[5]]:
        sentences["input"] = test_x
        feed_dict_map = {}
        for input_item in model_graph_signature.inputs.items():
            input_op_name = input_item[0]
            input_tensor_name = input_item[1].name
            feed_dict_map[input_tensor_name] = sentences[input_op_name]
        predict_y = session.run(output_tensor_names, feed_dict=feed_dict_map)
        print("predict pb y:",predict_y)

if __name__ == "__main__":
    save_model()
    session, m = load_model()
    export_model(session, m)
    load_pb()
