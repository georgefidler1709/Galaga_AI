import random
import retro
import time
import numpy as np
import tensorflow as tf
env = retro.make('GalagaDemonsOfDeath-Nes', '1Player.Level1')

GAMMA = 0.999 # discount factor, between 0 and 1, used in Bellman eq
INITIAL_EPSILON = 1 # starting value of epsilon, used in exploration
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 50 # decay period

STATE_DIM_1 = env.observation_space.shape[0]
STATE_DIM_2 = env.observation_space.shape[1]
STATE_DIM_3 = env.observation_space.shape[2]
# ACTION_OUTPUT_DIM is the length of the vector given to the env as an action
# ACTION_DIM is the largest meaningful action as a binary representation. Any higher will just duplicate an action represented by a lower number.
ACTION_OUTPUT_DIM = env.action_space.n
ACTION_DIM = 6

LIVES_PENALTY = 1000

epsilon = INITIAL_EPSILON

# ============================== Network ======================================

# --------- network inputs ----------
state_in = tf.placeholder("float", [None, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# --------- network hyperparameters ----------
fc1_units = 40
act = tf.tanh
init = tf.glorot_uniform_initializer()
lr=0.004
FILTER_SIZE = 100
FILTERS = 2
POOL_FILTER_SIZE = 50
POOL_STRIDE = 5
# --------------------------------------------
with tf.variable_scope('primary'):

    conv = tf.layers.conv2d(
        inputs=state_in,
        filters=FILTERS,
        kernel_size=FILTER_SIZE,
        activation=tf.nn.relu,
        padding='SAME',
        reuse=None
    )
    print("conv = " + str(conv.shape))

    #fc1 = tf.layers.dense(state_in, fc1_units, activation=act, kernel_initializer=init)
    # fc2 = tf.layers.dense(fc1, fc2_units, activation=act, kernel_initializer=init)
    # fc3 = tf.layers.dense(fc2, fc2_units, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pooled = tf.nn.max_pool(
        value=conv,
        ksize=[1, POOL_FILTER_SIZE, POOL_FILTER_SIZE, 1],
        strides=[1, POOL_STRIDE, POOL_STRIDE, 1],
        padding='SAME'
        )

    done_conv = tf.reshape(tensor=pooled, shape=[-1, (round(STATE_DIM_1/POOL_STRIDE) * round(STATE_DIM_2/POOL_STRIDE) * FILTERS)])
    #print("fc1 = " + str(fc1.shape))

    # TODO: Network outputs
    # output a q-value for EACH possible action, rank-1 tensor of length ACTION_DIM
    q_values = tf.layers.dense(done_conv, ACTION_DIM, kernel_initializer=init) # linear activation
    # extract the q-value of the action in action_in
    # by using action_in as a mask
    ones = tf.ones_like(action_in)
    action_bool = action_in >= ones
    action_bool = tf.cast(action_bool, dtype=tf.float32) 
    print("q_values = " + str(q_values.get_shape()))
    print("action_bool = " + str(action_bool.get_shape()))
    q_action = tf.reduce_sum(tf.multiply(action_bool, q_values), reduction_indices=1)

    # action_chosen = tf.argmax(q_action)

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
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })

    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)

    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

BATCH_SIZE = 120
MAX_MEM_SIZE = 30000 # WARNING prob want this to be smaller to be effective
TUPLE_DIM = 4 # each sample is a tupdle of (state, action, reward, next_state)
UPDATE_FREQ = 4 # TODO tune this for higher to make more stable

memory = []
# state is a tuple of (state, action, reward, next_state)
def add_step_to_memory(state):
    #print(state)
    memory.append(state)
    if(len(memory) > MAX_MEM_SIZE):
        memory.pop(0)

def get_batch_from_memory(batch_size):
    sample = random.sample(memory, batch_size)            
    return sample

attempt_num = 0

while True:
    t = 0
    done = False
    overall_score = 0
    lives = 2
    attempt_num += 1

    if epsilon > FINAL_EPSILON and attempt_num != 1:
            epsilon -= epsilon / EPSILON_DECAY_STEPS

    state = env.reset()

    while not done:
        print("attempt = " + str(attempt_num) + " t = " + str(t))
        action = explore(state, epsilon)
        action_input = np.array(list(format(np.argmax(action), '0' + str(ACTION_OUTPUT_DIM) +'b')), dtype=np.float32)
        next_state, _, done, info = env.step(action_input)

        if done:
            next_state = None

        reward = info['score'] - overall_score
        overall_score = info['score']
        #if info['lives'] != lives:
        #   reward = reward + (info['lives'] - lives) * LIVES_PENALTY
        #   lives = info['lives']    
        t += 1

        add_step_to_memory((state, action, reward, next_state))

        # perform training update after collecting some experience
        if (t != 0 and len(memory) > BATCH_SIZE) and (done or (t % UPDATE_FREQ == 0)):

            # sample random minibatch of transitions
            batch = get_batch_from_memory(BATCH_SIZE)

            batch_states = np.reshape(np.array([ x[0] for x in batch ]), [BATCH_SIZE, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])
            batch_actions = np.reshape(np.array([ x[1] for x in batch ]), [BATCH_SIZE, ACTION_DIM])
            batch_rewards = np.reshape(np.array([ x[2] for x in batch ]), [BATCH_SIZE, 1])
            batch_nexts = np.reshape(np.array([(np.zeros(STATE_DIM)
                                 if val[3] is None else val[3]) for val in batch ]), [BATCH_SIZE, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])

            curr_q_values = q_values.eval(feed_dict={
                state_in: batch_states
            })

            next_state_q_values = q_values.eval(feed_dict={
                 state_in: batch_nexts
            })

            next_q_vals_primary = q_values.eval(feed_dict={
                state_in: batch_nexts
            })

            # prepare array inputs needed to optimize
            targets = np.zeros(BATCH_SIZE)

            for i, sample in enumerate(batch):
                s_state, s_action, s_reward, s_next = sample[0], sample[1], sample[2], sample[3]

                if s_next is None:
                    targets[i] = s_reward
                else:
                    targets[i] = s_reward + GAMMA * np.max(next_state_q_values[i])
            print("t = " + str(t))
            print("targets = " + str(targets.shape))
            print("batch_actions = " + str(batch_actions.shape))
            print("batch_states = " + str(batch_states.shape))
            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: targets, # ERROR this shape might be weird if you use commented out version
                action_in: batch_actions,
                state_in: batch_states
            })

        # Update
        state = next_state

    print("attempt: {0}, score: {1}", attempt_num, overall_score)

env.close()