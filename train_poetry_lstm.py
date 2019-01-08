# encoding : utf-8
import tensorflow as tf
import numpy as np
import collections

#-----------数据处理----------------#
poetry_file = 'data/poetry.txt'

#诗集
poetrys = []
with open(poetry_file, "r") as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ','')
            #去除诗句中的一些停用词，去掉长度过长或者过短的诗句，并将古诗内容[]起来,将content放入poetrys列表中
            if '_' in content or '(' in content or '（' in content or '《' in content or '['  in content:
                continue
            if len(content) < 5 or len(content) > 79: #120
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass
# print(poetrys)
#按诗的内容字数排序
poetrys = sorted(poetrys, key=lambda line:len(line))
# print(poetrys)
print("诗总数：", len(poetrys), poetrys)

#统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
# print(counter)
count_pairs = sorted(counter.items(), key=lambda x:-x[1]) #按每一项的后面的出现次数值进行排序
print(count_pairs)
words, _ = zip(*count_pairs)

#取前多少个常用字
words = words[:len(words)] + (' ',)
#每个字映射成一个数字id
word_num_map = dict(zip(words, range(len(words))))
#把诗转换成向量形式
to_num = lambda word:word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]

#每次取2首诗来训练
batch_size = 64 #2
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size

    batches = poetrys_vector[start_index:end_index]
    length = max(map(len,batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:,:-1] = xdata[:,1:]

    x_batches.append(xdata)
    y_batches.append(ydata)


#---------------------------RNN-------------------------------
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])

#定义RNN
def neural_network(model='lstm', rnn_size=160, num_layers=2):
    if model == 'rnn':
        cell_fun = tf.nn.rnn_cell.BasicRNNCell
    elif model =='gru':
        cell_fun = tf.nn.rnn_cell.GRUCell
    elif model == 'lstm':
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope('rnnlm'):
        softmax_W = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
        softmax_b = tf.get_variable("softmax_b", [len(words)+1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])
    # tf.nn.elu()
    # tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    logits = tf.matmul(output, softmax_W) + softmax_b
    probs = tf.nn.softmax(logits)  #经过激活函数后的预测
    return logits, last_state, probs, cell, initial_state

#训练
def train_neural_network():
    logits, last_state, _, _, _ = neural_network()
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                                                              [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    # 获取所有可训练的向量
    tvars = tf.trainable_variables()
    #clip_by_norm是指对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题
    # clip_by_global_norm是梯度缩放输入是所有trainable向量的梯度，和所有trainable向量，返回第一个clip好的梯度
    #这个函数返回截取过的梯度张量和一个所有张量的全局范数。clip_norm 是截取的比率
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #tf.train.AdadeltaOptimizer(learning_rate)
    tf.train.AdadeltaOptimizer(learning_rate)
    #tf.train.Gra
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3) #tf.all_Varibles()
        for epoch in range(500): #70
            sess.run(tf.assign(learning_rate, 0.01*(0.97 ** epoch)))#0.018*(1.0 ** epoch)
            n = 0
            for batche in range(n_chunk):
                train_loss, _, _  = sess.run([cost, last_state, train_op],
                                             feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                n +=1
                print(epoch, batche, train_loss)
            if epoch % 7 == 0:
                 saver.save(sess, './module/poetry.ckpt', global_step=epoch)

train_neural_network()
