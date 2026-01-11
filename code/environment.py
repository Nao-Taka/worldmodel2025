import random

import numpy as np
import cv2
from termcolor import cprint

import genesis as gs
import torch

#########################
#実験用の環境
#ボールタッチタスク
#ボールに触ることで終了
#ボールに触るまでは0、触ったあとはずっと1
#200frameで終了
#ランダムで床から始まるのと
#空中から始まる2パターン
#ボールの初期位置はランダム

class BallTouchEnv():
    def __init__(self, n_envs=1, robot_height=0.3, ballfall=True):
        ################
        #setting for config
        self.n_step = 0
        self.ballfall = ballfall
        if ballfall:
            robot_height=0.3
        else:
            robot_height=0

        gs.init()
        # set scene
        self.scene = gs.Scene(
            show_viewer=True,
            sim_options=gs.options.SimOptions(
                dt=1e-2,
                substeps=4
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame = True,
                show_cameras = False
            ),
            rigid_options=gs.options.RigidOptions(
                enable_adjacent_collision=True,
            ),
            renderer=gs.renderers.BatchRenderer(),
            )
        
        # add camera
        self.cam = self.scene.add_camera(
            res=(800, 600),
            pos=(-1.3, 0, robot_height+0.6),
            lookat=(0., 0, robot_height+0.3),
            GUI=True,
        )
        #add light
        light = self.scene.add_light(
            pos=(0, 0, 100),
            dir=(0, 0, -1),

        )

        ###################
        ##Entities
        # floor
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(
                coup_restitution=1,
                friction=0.3,
                sdf_cell_size=2e-3
            )
            )
        
        # object
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=self.__random_x_y(),
                radius=0.02
                ),
            material=gs.materials.Rigid(
                rho=1000,
                coup_restitution=1,
                friction=0.1,
                sdf_cell_size=2e-3
            ),
            visualize_contact=True,
        )

        # robot
        self.so101 = self.scene.add_entity(
            gs.morphs.MJCF(
                file='SO-ARM100/Simulation/SO101/so101_new_calib.xml',
                pos=(0, 0, robot_height),
                ),
        )

        ################
        #build
        if n_envs is not None:
            self.n_envs = n_envs
            self.scene.build(n_envs=n_envs, env_spacing=(1,1))
        else:
            self.scene.build()

        ################
        #setting for action
        jnt_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
            'gripper'
        ]
        self.dofs_idx = [self.so101.get_joint(name).dof_idx_local for name in jnt_names]

        self.reset()

    def reset(self):
        self.n_step = 0
        self.scene.reset()
        for b in range(self.n_envs):
            self.ball.set_pos(self.__random_x_y(ballfall=self.ballfall), envs_idx=b)
            # self.ball.set_pos((0.4, 0, 1), envs_idx=b)
        rgb,_,_,_ = self.scene.render_all_cameras()
        obs = rgb[0].detach().cpu().numpy()
        return obs


    def step(self, action):
        '''
        action:np.array[n_envs, 6]
        '''
        info = {}
        #simulation
        force = torch.as_tensor(action, dtype=torch.float32)
        self.so101.control_dofs_position(
            force,
            dofs_idx_local=self.dofs_idx
        )
        self.scene.step()
        self.n_step += 1
        rgb,_,_,_ = self.scene.render_all_cameras()



        obs = rgb[0].detach().cpu().numpy()

        info.update(self.__check_touch_ball_to_gripper())

        done = self.__check_done(info)
        reward = self.__calc_reward(info) #temporary

        #truncated check
        truncated = False
        if self.n_step > 200:
            truncated = True
            
        return obs, reward, done, truncated, info
    
    def render(self):
        pass
    
  
    def __check_touch_ball_to_gripper(self):
        # is_touch_a = torch.zeros(self.n_envs, dtypoe=torch.bool)
        # is_touch_b = torch.zeros(self.n_envs, dtypoe=torch.bool)
        #ボールと接触しているものを選ぶ
        contacts_info = self.ball.get_contacts()
        contacts_info_lnkA = contacts_info['link_a']
        contacts_info_lnkB = contacts_info['link_b']
        #linkAのi番目にsphere_baselinkのインデックスが含まれるか
        #linkBにも同様のインデックスが含まれるか
        #マスク画像を作成し、各i番目ごとに比較
        mask_a   = (contacts_info_lnkA == self.ball.get_link('sphere_baselink').idx)
        mask_b_1 = (contacts_info_lnkB == self.so101.get_link('gripper').idx)
        mask_b_2 = (contacts_info_lnkB == self.so101.get_link('moving_jaw_so101_v1').idx)
        #ボールがグリッパーにふれたかどうか
        is_touch_a = torch.any(mask_a & mask_b_1, dim=1)
        is_touch_b = torch.any(mask_a & mask_b_2, dim=1)
        return {"is_touch_to_gripper_a":is_touch_a,
                "is_touch_to_gripper_b":is_touch_b}
        if torch.any(contacts_info_lnkB==7, dim=1):
            is_touch_a = True
        if torch.any(contacts_info_lnkB==8, dim=1):
            is_touch_b = True
        return is_touch_a, is_touch_b
        
    def __check_done(self, info):
        is_gripper_a = info['is_touch_to_gripper_a']
        is_gripper_b = info['is_touch_to_gripper_b']
        ret = is_gripper_a & is_gripper_b
        ret = ret.cpu().numpy().astype('bool')
        return ret
        pass

    def __calc_reward(self, info):
        is_gripper_a = info['is_touch_to_gripper_a']
        is_gripper_b = info['is_touch_to_gripper_b']
        ret = (is_gripper_a.int() + is_gripper_b.int())/2.0
        ret = ret.cpu().numpy().astype(np.float32)
        return ret
    
    
    def __random_x_y(self, ballfall=True):
        '''
        ボールを落とす場所を決める
        '''
        theta = random.uniform(-1, 1)
        r = random.uniform(0.2, 0.6)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        if ballfall:
            return tuple(np.array((x, y, 1)))
        else:
            return tuple(np.array((x, y, 0.02)))
            


if __name__=='__main__':
    env = BallTouchEnv(n_envs=10, ballfall=True)
    for i in range(1000):
        if i%300==0:
            env.reset()
        obs, reward, done, truncated, info = env.step(np.array([0,1,-1,0,1.5,0.1]))
        print(reward)
        print(done)
        print(truncated)
        # for i in range(10):
        #     cv2.imshow(f'camera{i}', cv2.cvtColor(obs[i], cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        # print(reward)