import tensorflow as tf


class LREventClassifier:
    def __init__(self, vocab_size, learning_rate, unlbreg_lambda, l2reg_lambda):
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.unlbreg_lambda = unlbreg_lambda
        self.l2reg_lambda = l2reg_lambda
        self.thetaEW = tf.Variable(tf.random_normal([1, vocab_size], mean=0.0, stddev=0.5, dtype=tf.float32),
                                   name='event_thetaEW')
        self.thetaEb = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='event_thetaEb')
        self.params = [self.thetaEW, self.thetaEb]
        self.construct_graph(vocab_size, learning_rate, unlbreg_lambda, l2reg_lambda)
    
    def construct_graph(self, vocab_size=None, learning_rate=None, unlbreg_lambda=None, l2reg_lambda=None):
        vocab_size = vocab_size if vocab_size is not None else self.vocab_size
        learning_rate = learning_rate if learning_rate is not None else self.learning_rate
        unlbreg_lambda = unlbreg_lambda if unlbreg_lambda is not None else self.unlbreg_lambda
        l2reg_lambda = l2reg_lambda if l2reg_lambda is not None else self.l2reg_lambda
        
        self.seedxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.seedye = tf.placeholder(tf.float32, [None, 1])
        self.seedscore = tf.nn.xw_plus_b(self.seedxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.seedpred = tf.sigmoid(self.seedscore)
        self.seedcross = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.seedscore, labels=self.seedye)
        self.seedloss = tf.reduce_mean(self.seedcross)
        
        self.unlbxe = tf.placeholder(tf.float32, [None, vocab_size])
        self.unlbye = tf.placeholder(tf.float32, [1, 1])
        self.unlbscore = tf.nn.xw_plus_b(self.unlbxe, tf.transpose(self.thetaEW), self.thetaEb)
        self.unlbpred = tf.sigmoid(self.unlbscore)
        self.unlbpredave = tf.reduce_mean(self.unlbpred)
        self.unlbcross = self.cross_entropy(self.unlbpredave, self.unlbye) - \
                         self.cross_entropy(self.unlbye, self.unlbye)
        self.unlbloss = self.unlbcross
        
        self.unlbreg_lambda = unlbreg_lambda
        self.l2reg_lambda = l2reg_lambda
        self.l2reg = tf.nn.l2_loss(self.thetaEW) + tf.nn.l2_loss(self.thetaEb)
        # self.loss = self.seedloss + self.l2reg_lambda * self.l2reg
        self.loss = self.seedloss + self.unlbreg_lambda * self.unlbloss + self.l2reg_lambda * self.l2reg
        self.trainop = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def cross_entropy(self, logits, labels):
        """tf.nn.sigmoid_cross_entropy_with_logits(logits=a, labels=b)
           makes log(sigmoid(a)), and then cross it with its label b"""
        log_loss = labels * tf.log(logits) + (tf.constant(1, dtype=tf.float32) - labels) * tf.log(1 - logits)
        return - log_loss
    
    def train_per_step(self, sx, sy, ux, uy):
        _, loss = self.sess.run(fetches=[self.trainop, self.loss],
                                feed_dict={self.seedxe: sx, self.seedye: sy, self.unlbxe: ux,
                                           self.unlbye: uy})
        return loss
    
    def predict(self, idfmtx):
        return self.sess.run(self.seedpred, feed_dict={self.seedxe: idfmtx})
    
    def get_theta(self):
        return self.sess.run(self.params)
    
    def load_params(self, file_name):
        saver = tf.train.Saver(self.params)
        saver.restore(self.sess, file_name)
    
    def save_params(self, file_name):
        saver = tf.train.Saver(self.params)
        saver.save(self.sess, file_name)


class UnknownTypeEventClassifier(LREventClassifier):
    def __init__(self, vocab_size, learning_rate, unlbreg_lambda, l2reg_lambda):
        LREventClassifier.__init__(self, vocab_size, learning_rate, unlbreg_lambda, l2reg_lambda)
