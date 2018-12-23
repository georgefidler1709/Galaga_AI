import random
import retro
import time
import numpy as np
import tensorflow as tf
env = retro.make(game='GalagaDemonsOfDeath-Nes', state='1Player.Level1', record='.')

GAMMA = 0.999 # discount factor, between 0 and 1, used in Bellman eq
INITIAL_EPSILON = 1 # starting value of epsilon, used in exploration
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 75 # decay period
FINAL_EPISODE = 300

MAX_STEPS = 15000

STATE_DIM_1 = env.observation_space.shape[0]
STATE_DIM_2 = env.observation_space.shape[1]
STATE_DIM_3 = env.observation_space.shape[2]
# ACTION_OUTPUT_DIM is the length of the vector given to the env as an action
# ACTION_DIM is the largest meaningful action as a binary representation. Any higher will just duplicate an action represented by a lower number.
ACTION_OUTPUT_DIM = env.action_space.n
ACTION_DIM = 6

epsilon = INITIAL_EPSILON

# ============================== Network ======================================

# --------- network inputs ----------
state_in = tf.placeholder("float", [None, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# --------- network hyperparameters ----------
act = tf.tanh
init = tf.glorot_uniform_initializer()
lr=0.004

FILTER_SIZE1 = 16
FILTERS1 = 16
FILTER_STRIDE1 = 4

FILTER_SIZE2 = 8
FILTERS2 = 32
FILTER_STRIDE2 = 2

POOL_FILTER_SIZE = 8
POOL_STRIDE = 2

fc1_units = 256

# --------------------------------------------
with tf.variable_scope('primary'):
    print("state_in = " + str(state_in.shape))
    conv1 = tf.layers.conv2d(
        inputs=state_in,
        filters=FILTERS1,
        kernel_size=FILTER_SIZE1,
        activation=tf.nn.relu,
        padding='VALID',
        strides=FILTER_STRIDE1
    )
    print("conv1 = " + str(conv1.shape))

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=FILTERS2,
        kernel_size=FILTER_SIZE2,
        activation=tf.nn.relu,
        padding='VALID',
        strides=FILTER_STRIDE2
    )
    print("conv2 = " + str(conv2.shape))

    pooled = tf.nn.max_pool(
        value=conv2,
        ksize=[1, POOL_FILTER_SIZE, POOL_FILTER_SIZE, 1],
        strides=[1, POOL_STRIDE, POOL_STRIDE, 1],
        padding='SAME'
        )
    print("pooled = " + str(pooled.shape))
    # fc2 = tf.layers.dense(fc1, fc2_units, activation=act, kernel_initializer=init)
    # fc3 = tf.layers.dense(fc2, fc2_units, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    done_conv = tf.layers.flatten(pooled)
    # done_fc1 = tf.reshape(tensor=fc1, shape= [-1, STATE_DIM_1 * STATE_DIM_2 * fc1_units])

    fc1 = tf.layers.dense(done_conv, fc1_units, activation=act, kernel_initializer=init)
    #print("fc1 = " + str(fc1.shape))

    # TODO: Network outputs
    # output a q-value for EACH possible action, rank-1 tensor of length ACTION_DIM

    #q_values = tf.layers.dense(done_conv, ACTION_DIM, kernel_initializer=init) # linear activation
    q_values = tf.layers.dense(fc1, ACTION_DIM, kernel_initializer=init) # linear activation

    # extract the q-value of the action in action_in
    # by using action_in as a mask
    ones = tf.ones_like(action_in)
    action_bool = action_in >= ones
    action_bool = tf.cast(action_bool, dtype=tf.float32) 
    # print("q_values = " + str(q_values.get_shape()))
    # print("action_bool = " + str(action_bool.get_shape()))
    q_action = tf.reduce_sum(tf.multiply(action_bool, q_values), reduction_indices=1)

    # TODO: Loss/Optimizer Definition
    # should be a function of target_in and q_action
    loss = tf.reduce_sum(tf.square(target_in - q_action))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

# ============================== Code ======================================


def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """


    # print(Q_estimates)
    # print(np.argmax(Q_estimates))
    # print()   

    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        Q_estimates = q_values.eval(feed_dict={
            state_in: [state]
            })
        with open("log.txt", 'a') as log:
            log.write(str(Q_estimates) + '\n')
        action = np.argmax(Q_estimates)

    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

BATCH_SIZE = 64
MAX_MEM_SIZE = 20000 # WARNING prob want this to be smaller to be effective
TUPLE_DIM = 4 # each sample is a tuple of (state, action, reward, next_state)
UPDATE_FREQ = 5 # TODO tune this for higher to make more stable

memory = []
save_scoring = [0] * MAX_MEM_SIZE
# state is a tuple of (state, action, reward, next_state)
def add_step_to_memory(state):
    memory.append(state)
    if(len(memory) > MAX_MEM_SIZE):
        to_remove = random.randint(0, MAX_MEM_SIZE - 1)
        while (memory[to_remove][2] != 0 and save_scoring[to_remove] == 0):
            save_scoring[to_remove] = 1
            to_remove = random.randint(0, MAX_MEM_SIZE - 1)
        save_scoring[to_remove] = 0
        memory.pop(to_remove)

def get_batch_from_memory(batch_size):
    sample = random.sample(memory, batch_size)            
    return sample

saver = tf.train.Saver()
episode = 0
while epsilon > FINAL_EPSILON and episode <= FINAL_EPISODE:
    t = 0
    done = False
    overall_score = 0
    lives = 2

    if epsilon > FINAL_EPSILON and episode != 0:
            epsilon -= epsilon / EPSILON_DECAY_STEPS

    state = env.reset()

    while t <= MAX_STEPS and not done:
        #env.render()
        t += 1
        action = explore(state, epsilon)
        action_input = np.array(list(format(np.argmax(action), '0' + str(ACTION_OUTPUT_DIM) +'b')), dtype=np.float32)
        # print("action = " + str(action))
        # print("action-input = " + str(action_input))
        #time.sleep(10)
        # print()
        next_state, _, done, info = env.step(action_input)

        if done:
            next_state = None

        if done or t == MAX_STEPS:
            # next_state = None
            reward = info['score']
        else:
            reward = 0

        add_step_to_memory((state, action, reward, next_state))

        # perform training update after collecting some experience
        if (t != 0 and len(memory) > BATCH_SIZE) and (done or (t % UPDATE_FREQ == 0)):
            with open("log.txt", 'a') as log:
                log.write("attempt = " + str(episode) + " t = " + str(t) + '\n')
            # sample random minibatch of transitions
            batch = get_batch_from_memory(BATCH_SIZE)
            batch_states = np.reshape(np.array([ x[0] for x in batch ]), [BATCH_SIZE, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])
            batch_actions = np.reshape(np.array([ x[1] for x in batch ]), [BATCH_SIZE, ACTION_DIM])
            batch_rewards = np.reshape(np.array([ x[2] for x in batch ]), [BATCH_SIZE, 1])
            batch_nexts = np.reshape(np.array([(np.zeros([STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])
                                 if x[3] is None else x[3]) for x in batch ]), [BATCH_SIZE, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])

            curr_q_values = q_values.eval(feed_dict={
                state_in: batch_states
            })

            next_state_q_values = q_values.eval(feed_dict={
                 state_in: batch_nexts
            })

            # prepare array inputs needed to optimize
            targets = np.zeros(BATCH_SIZE)

            for i, sample in enumerate(batch):
                s_state, s_action, s_reward, s_next = sample[0], sample[1], sample[2], sample[3]
                #print(t)
                #print("s_reward = " + str(s_reward))
                #print("next_q_max = " + str(next_state_q_values[i]))
                #print()

                if s_next is None:
                    targets[i] = s_reward
                else:
                    targets[i] = s_reward + GAMMA * np.max(next_state_q_values[i])

            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: targets,
                action_in: batch_actions,
                state_in: batch_states
            })

        # Update
        state = next_state

    with open("log.txt", 'a') as log:
        log.write("attempt: %d, score: %d, epsilon: %.2lf\n" % (episode, overall_score, epsilon))
        if episode % 5 == 0:
            save_path = saver.save(session, "./models/model_" + str(episode) +".ckpt")
            log.write("Model Saved: episode %d\n" % (episode))
    episode += 1

env.close()
