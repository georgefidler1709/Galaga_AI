import random
import retro
import time
import numpy as np
import tensorflow as tf
from skimage import color, transform
from collections import deque

env = retro.make(game='GalagaDemonsOfDeath-Nes', state='1Player.Level1', record='.')

GAMMA = 0.9 # discount factor, between 0 and 1, used in Bellman eq
INITIAL_EPSILON = 1 # starting value of epsilon, used in exploration
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period

MAX_STEPS = 5000

STATE_DIM_1 = 110
STATE_DIM_2 = 84
STATE_DIM_3 = 4
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
act = tf.nn.elu
init = tf.glorot_uniform_initializer()
lr=0.00025

FILTER_SIZE1 = 8
FILTERS1 = 32
FILTER_STRIDE1 = 2

FILTER_SIZE2 = 4
FILTERS2 = 64
FILTER_STRIDE2 = 2

FILTER_SIZE3 = 3
FILTERS3 = 64
FILTER_STRIDE3 = 2

POOL_FILTER_SIZE = 8
POOL_STRIDE = 2

fc1_units = 512

# --------------------------------------------
with tf.variable_scope('primary'):
    print("state_in = " + str(state_in.shape))
    conv1 = tf.layers.conv2d(
        inputs=state_in,
        filters=FILTERS1,
        kernel_size=FILTER_SIZE1,
        activation=tf.nn.elu,
        padding='VALID',
        strides=FILTER_STRIDE1,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )
    print("conv1 = " + str(conv1.shape))

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=FILTERS2,
        kernel_size=FILTER_SIZE2,
        activation=tf.nn.elu,
        padding='VALID',
        strides=FILTER_STRIDE2,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )
    print("conv2 = " + str(conv2.shape))

    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=FILTERS3,
        kernel_size=FILTER_SIZE3,
        activation=tf.nn.elu,
        padding='VALID',
        strides=FILTER_STRIDE3,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
    )
    print("conv3 = " + str(conv2.shape))

    # pooled = tf.nn.max_pool(
    #     value=conv2,
    #     ksize=[1, POOL_FILTER_SIZE, POOL_FILTER_SIZE, 1],
    #     strides=[1, POOL_STRIDE, POOL_STRIDE, 1],
    #     padding='SAME'
    #     )
    # print("pooled = " + str(pooled.shape))
    # fc2 = tf.layers.dense(fc1, fc2_units, activation=act, kernel_initializer=init)
    # fc3 = tf.layers.dense(fc2, fc2_units, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    done_conv = tf.layers.flatten(conv3)
    #done_conv = tf.layers.flatten(pooled)
    # done_fc1 = tf.reshape(tensor=fc1, shape= [-1, STATE_DIM_1 * STATE_DIM_2 * fc1_units])

    fc1 = tf.layers.dense(done_conv, fc1_units, activation=act, kernel_initializer=init)
    #print("fc1 = " + str(fc1.shape))

    # TODO: Network outputs
    # output a q-value for EACH possible action, rank-1 tensor of length ACTION_DIM

    #q_values = tf.layers.dense(done_conv, ACTION_DIM, kernel_initializer=init) # linear activation
    q_values = tf.layers.dense(fc1, ACTION_DIM, activation=None, kernel_initializer=init) # linear activation

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

#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb
def preprocess_frame(frame):
    # Greyscale frame 
    grey = color.rgb2gray(frame)    
    # Crop the screen to get rid of player information
    # [Up: Down, Left: right]
    cropped_frame = grey[8:-1,4:-48]
    normalised_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalised_frame, [STATE_DIM_1,STATE_DIM_2])
    
    return preprocessed_frame # 110x84x1 frame


STACK_SIZE = STATE_DIM_3 # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames  =  deque([np.zeros((STATE_DIM_1, STATE_DIM_2), dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((STATE_DIM_1, STATE_DIM_2), dtype=np.int) for i in range(STACK_SIZE)], maxlen=STACK_SIZE)
        
        # Because we're in a new episode, copy the same frame 4x
        for i in range(STACK_SIZE):
            stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """

    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    with open("log.txt", 'a') as log:
        log.write(str(Q_estimates) + '\n')

    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)

    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

BATCH_SIZE = 64
MAX_MEM_SIZE = 20000 # WARNING prob want this to be smaller to be effective
UNUSUAL_SAMPLE_PRIORITY = 0.99
TUPLE_DIM = 4 # each sample is a tuple of (state, action, reward, next_state)
UPDATE_FREQ = 5 # TODO tune this for higher to make more stable

memory = []
# state is a tuple of (state, action, reward, next_state)
def add_step_to_memory(state):
    memory.append(state)
    if(len(memory) > MAX_MEM_SIZE):
        memory.pop(0)

# batch collection updated so that scoring states are prioritised over non-scoring states probabilistically
def get_batch_from_memory(batch_size):
    buffer = [(i, x[2]) for i, x in enumerate(memory)]
    buffer = sorted(buffer, key=lambda replay: abs(replay[1]), reverse=True)
    p = np.array([UNUSUAL_SAMPLE_PRIORITY ** i for i in range(len(memory))])
    p = p / sum(p)
    sample_idxs = np.random.choice(np.arange(len(memory)),size=batch_size, p=p, replace=False)
    sample = [memory[buffer[idx][0]] for idx in sample_idxs]
    return sample
    
    # sample = random.sample(memory, batch_size)            
    # return sample

saver = tf.train.Saver()

avg_reward_scaled = 0
tot_reward_scaled = 0
episode = 0
while epsilon > FINAL_EPSILON:
    t = 0
    done = False
    overall_score = 0

    if epsilon > FINAL_EPSILON and episode != 0:
            epsilon -= epsilon / EPSILON_DECAY_STEPS

    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, True)

    while t <= MAX_STEPS and not done:
        #env.render()
        t += 1
        action = explore(state, epsilon)
        action_input = np.array(list(format(np.argmax(action), '0' + str(ACTION_OUTPUT_DIM) +'b')), dtype=np.float32)
        next_state, _, done, info = env.step(action_input)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        if done or t == MAX_STEPS:
            next_state = np.array(np.zeros([STATE_DIM_1, STATE_DIM_2, STATE_DIM_3]))
            # reward based solely on final score achieved in this episode
            overall_score = info['score']
            # reward normalised around the average score achieved to aid in learning
            reward = np.tanh((overall_score / 1000) - avg_reward_scaled)
            tot_reward_scaled += (overall_score / 1000)
            avg_reward_scaled = (tot_reward_scaled / (episode + 1))
        else:
            reward = 0

        # steps only added to memory if they are interesting or at set intervals to  promote a variety of samples in memory
        #if reward != 0 or done or (t % UPDATE_FREQ == 0):
        add_step_to_memory((state, action, reward, next_state))

        # perform training update after collecting some experience
        if (episode != 0 and len(memory) > BATCH_SIZE) and (done or (t % UPDATE_FREQ == 0)):
            with open("log.txt", 'a') as log:
                log.write("attempt = " + str(episode) + " t = " + str(t) + '\n')
            # sample random minibatch of transitions
            batch = get_batch_from_memory(BATCH_SIZE)
            batch_states = np.reshape(np.array([ x[0] for x in batch ]), [BATCH_SIZE, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])
            batch_actions = np.reshape(np.array([ x[1] for x in batch ]), [BATCH_SIZE, ACTION_DIM])
            batch_rewards = np.reshape(np.array([ x[2] for x in batch ]), [BATCH_SIZE, 1])
            batch_nexts = np.reshape(np.array([ x[3] for x in batch ]), [BATCH_SIZE, STATE_DIM_1, STATE_DIM_2, STATE_DIM_3])

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
        log.write("reward: %d, tot_reward: %d, avg_reward: %d\n" % (reward, tot_reward_scaled, avg_reward_scaled))
        if episode % 5 == 0:
            save_path = saver.save(session, "./models/model_" + str(episode) +".ckpt")
            log.write("Model Saved: episode %d\n" % (episode))
    episode += 1

env.close()
