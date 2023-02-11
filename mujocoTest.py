#!/usr/bin/env python3

import mujoco
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100) 

import mediapy as media
from matplotlib import pyplot as plt 

# Testing Mujoco Library

# https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=eU7uWNsTwmcZ

xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)

model.ngeom
print(model.ngeom)
model.geom_rgba
print(model.geom_rgba)

try:
  model.geom()
except KeyError as e:
  print(e)

model.geom('green_sphere')
print(model.geom('green_sphere'))

model.geom('green_sphere').rgba
print(model.geom('green_sphere').rgba)

id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'green_sphere')
model.geom_rgba[id, :]

print(model.geom_rgba[id, :])

print('id of "green_sphere": ', model.geom('green_sphere').id)
print('name of geom 1: ', model.geom(1).name)
print('name of body 0: ', model.body(0).name)

[model.geom(i).name for i in range(model.ngeom)]
print([model.geom(i).name for i in range(model.ngeom)])

data = mujoco.MjData(model)
print(data)

print(data.geom_xpos)

mujoco.mj_kinematics(model, data)
print('raw access:\n', data.geom_xpos)

# MjData also supports named access:
print('\nnamed access:\n', data.geom('green_sphere').xpos)

# Basic rendering, simulation, and animation

# Make model and data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Make renderer, render and show the pixels
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model)
print(renderer)
print('rendered object:',  renderer.__dir__())
print(type(renderer))

'''
while True:
    media.show_image(renderer.render())
'''