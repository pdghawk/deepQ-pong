import DQN.qnet as qnet
import numpy as np
import gym

game = 'Pong-v0'
env = gym.make(game)
frame = env.reset()

def preprocess(frame):
      frame_out = np.zeros((84,84),dtype=np.uint8)
      # to black and white
      tmp = np.mean(frame, axis=2)
      # trim edges
      tmp = tmp[28:-12, :]
      # downsample
      tmp = tmp[1:-1:2,::2]
      frame_out[:,2:-2] = tmp.astype(np.uint8)
      return frame_out

frame=preprocess(frame)
N_x=np.size(frame,0)
N_y=np.size(frame,1)

mynet=qnet.TripleConvQnet()
mynet.set_dimensions(N_x,N_y,4,3)

obs=np.zeros((N_x,N_y,4))
obs[:,:,0]=frame
for i in range(1,4):
    f,_,_,_=env.step(0)
    obs[:,:,i]=preprocess(f)

zout = mynet.run(obs,'online/')

print(zout)
