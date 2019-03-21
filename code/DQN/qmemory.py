import numpy as np

class Qmemory:
    """ memory object for Q learning

    This class provides a way to define a memory object for depp Q learning,
    to provide memory recall. memory is made up of memory of transitions during
    'play'. This is made of memory of states i, j (=i+1), the action take at i (to get to j)
    the reward earned by making that action, and whether the state j was the end of the game.

    transitions can be added to the memory with the 'write' method. This will automatically
    add the transition data to the memory, and once the memory is full, will start
    to rewrite the memory of events from many moves ago.

    One can also retrieve a subset of the memory using 'get_batch'. This selects a
    random subset, taking into account whether the memory is full, and if not only pulling
    from examples that are already 'filled in' in the memory.
    """
    def __init__(self,N_mem, N_x, N_y, N_frames):
        """ Create a memory object

        Create an object for storing transitions between states in the game/process
        being learned. A single transition contains (phi_i,phi_j,a_i,r_i,terminal_i):

        phi_i: state before action taken
        phi_j: state after action taken
        a_i:   the action that was taken
        r_i:   the reward received for this action
        terminal_i: whether this move was terminal for the game/process

        Args:
            N_mem: how large the memory should be = how many transitions to store
            obs_data_size: how large is each observation (state), as a single integer.
                        e.g a 2d frame of 80x80 80x80=1600
            N_frames: how many frames are stored per state
        """
        self.N_mem         = N_mem
        #self.obs_data_size = obs_data_size
        self.N_frames      = N_frames

        # initialize the memory arrays.
        self.memory_phi_i      = np.zeros((N_mem,N_x,N_y,N_frames),dtype=np.uint8)
        self.memory_phi_j      = np.zeros((N_mem,N_x,N_y,N_frames),dtype=np.uint8)
        self.memory_a_i        = np.zeros((N_mem,1),dtype=np.uint8)
        self.memory_r_i        = np.zeros((N_mem,1),dtype=np.float32)
        self.memory_terminal_i = np.zeros((N_mem,1),dtype=np.float32)

        self.mem_count = 0

    def write(self,phi_i,phi_j,action,reward,terminal_float):
        """ Write a set of transition data to the memory

        Args:
            phi_i: state i
            phi_j: state_j
            action: action taken at i
            reward: reward recieved
            terminal_float: is the move terminal, is it the last move in game, as a float: 0 means is last move, 1 means isnt.

        writes the data into position np.mod(self.mem_count,self.N_mem) in the memory.
        This means that it will loop back to position 0 once the memory is full, and
        memory will be rewritten.
        """
        self.memory_phi_i[np.mod(self.mem_count,self.N_mem),:,:,:] = phi_i
        self.memory_phi_j[np.mod(self.mem_count,self.N_mem),:,:,:] = phi_j
        self.memory_a_i[np.mod(self.mem_count,self.N_mem),:] = action.astype(np.uint8)
        self.memory_r_i[np.mod(self.mem_count,self.N_mem),:] = reward
        self.memory_terminal_i[np.mod(self.mem_count,self.N_mem),:] = terminal_float
        # we just added some data, updtae our counter to tell us how manyh we added in total
        self.mem_count+=1

    def get_batch(self,N_get):
        """ Get a subset of the memory for training Q network.

        Args:
            N_get: how many transition event to get (i.e. return)

        Returns:
            batch_dict: a dictionary containing memory arrays:
                        - phi_i: state i
                        - phi_j: state j (=i+1)
                        - r_i: reward
                        - t_i: whether terminal
        """
        # check if memory is full or not

        if self.mem_count>=self.N_mem:
            # is full
            max_val = self.N_mem
        else:
            # isn't full - max index to look up to is the current count
            max_val = self.mem_count

        # get random integeres between 0 and our max_val defined above
        rand_ints = np.random.randint(0,high=max_val,size=N_get)

        # use rand_ints to get random memory selection
        batch_dict = {'phi_i': self.memory_phi_i[rand_ints,:,:,:],
                      'phi_j': self.memory_phi_j[rand_ints,:,:,:],
                      'a_i': self.memory_a_i[rand_ints,:],
                      'r_i': self.memory_r_i[rand_ints,:],
                      't_i': self.memory_terminal_i[rand_ints,:]}
        return batch_dict
