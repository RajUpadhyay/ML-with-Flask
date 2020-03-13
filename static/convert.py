from tensorflow.compat.v1.keras import backend as K

from tensorflow import keras

import tensorflow as tf

model = keras.models.load_model("model.h5")

# Create, compile and train model...

import os, argparse

import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("/Users/samirupadhyaya/dog-cat-classification/flask-app/static/model.h5", type=str, default="", help="/Users/samirupadhyaya/dog-cat-classification/flask-app/static/")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)



frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs]) 

tf.train.write_graph(frozen_graph, "/Users/samirupadhyaya/dog-cat-classification/flask-app/static/", "my_model.pb", as_text=False)