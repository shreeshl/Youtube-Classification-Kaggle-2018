a = tf.convert_to_tensor(np.ndarray.astype(np.random.randn(10,300,1024), np.float32))
b = tf.convert_to_tensor(np.ndarray.astype(np.random.randn(10,300,1024), np.float32))

cell = tf.nn.rnn_cell.GRUCell(num_units=1024)

outputs1, states1  = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell,
    cell_bw=cell,
    dtype=tf.float32,
    inputs=a)


sess = tf.Session()  
init = tf.global_variables_initializer()
sess.run(init)
sess.run(outputs1).shape
