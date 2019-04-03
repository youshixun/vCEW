# -*- coding: utf-8 -*-
""""
Created on Thu Aug  9 19:54:15 2018 
@author: Radar_Jamming_Laboratory. 204-{Shixun You}

"""
import util 
import generate 
import viewerUI
import resources
# import recognization 

import math
import pyglet
import numpy as np
import numpy.random as npr

#from PIL import Image

MAP_CENTER = np.array([300, 300])

DIFFICULTY = [1, 0.5, 0.2, 0.1] # 0-3 LEVELS

MEMORY_CAPACITY = 300000

random_target = generate.playerGenerate(_type='Station', _id=1)

#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------  
class ElectronicWarfare_Env(object):
    
    "game environment based on 600 x (600+450) pixels, this version only supports 1 vs 1" 
    
    viewer = None
        
    stateDim = 9
    
    actionDim = 2
    
    name = None
              
    def __init__(self):
        
        "游戏成员初始化"
        
        self.winner = 'None'               
        
        self.in_range = 0
        
        self.initial_dis = 0
        
        self.players = {}        
        
        # attacker                
        self.players['attackers'] = [generate.playerGenerate(_type='UCAV', _id=1)]
        
        # defender
        self.players['defenders'] = [generate.playerGenerate(_type='Station', _id=1)]
        
        # map
        self.map = pyglet.sprite.Sprite(resources.map_image) 
        
    #--------------------------------------------------------------------------    
    def seed(self, num):
        
        np.random.seed(num)
        
    #--------------------------------------------------------------------------    
    def levels(self, K=None):
        
        if K is None: K = 0
        
        self.dt = DIFFICULTY[K]   
        
        self.name = 'Explorer_V1 (15 x 15 km2)_'+str(self.dt)
        
        return int(400/self.dt)
        
    #--------------------------------------------------------------------------            
    def reset(self, display=True, site=None):
        
        "游戏成员属性重置"
        
        self.in_range = 0         
         
        for player in self.players['attackers']+self.players['defenders']:
            
            if display:
                
                if self.viewer is not None: 
            
                    self.viewer.OperatingStep = 0
                    
                    self.winner = 'None'   
                
                player.re_init(site, batch=self.viewer.batch, group=self.viewer.foreground) 
             
            else:
            
                player.re_init(site)
            
            player.init_type(_type=player._type)  
            
            for ARM_ in player.kidsARM: ARM_[0].batch = None
                                 
        # electromagnetic spectrum environment
        Att, Def = self.players['attackers'][0], self.players['defenders'][0]
        
        dis, vec = util.displacement(Att.spherePos, Def.spherePos) 
        
        disp = np.array([dis*vec[0]/15, dis*vec[1]/15, (Def.h.real-Att.h.real) / 7.5])
        
        self.initial_dis = dis
        
        Att.cESE(disp)
        
        Att.path.append(Att.spherePos)
               
        return Att.pointState[-1]          
            
    #--------------------------------------------------------------------------       
    def step(self, action):
        
        "与环境交互"
        
        reward = 0
        
        done = True
        
        Att, Def = self.players['attackers'][0], self.players['defenders'][0]
                
        # decoding actions for attackers        
        angles = action*math.pi*[1, 0.5]
            
        aMov = util.unitV(angles)

        Att.pos_plan(self.dt, aMov)        
        
        for player in self.players['attackers']+self.players['defenders']:
                                    
            player.collision(self.players['attackers']+self.players['defenders']) # detecting the collision of UCAVs                            
        
        # 只用来评估作战胜利       
        if Def.is_dead: 
            
            self.winner = 'attacker'
        
            reward += 100 # win add 100
            
        else: 
            
            self.winner = 'None'
                
        # operation                       
        Def.operation(Att) 
                
        Def.checkingLib()
        
        dis_r, vec_r = Att.operation(Def) # real
            
        Att.checkingLib() 
        
        disp_r = np.array([dis_r*vec_r[0]/15, dis_r*vec_r[1]/15, (Def.h.real-Att.h.real) / 7.5]) # real
        
        # reward shaping             
        angle_behavior = math.acos(aMov.dot(vec_r))/math.pi
        
        reward = -np.linalg.norm(disp_r)
        
        reward += self.search_reward(Att)
        
        if dis_r < Att.radar.dR['Search']: # capture
            
            reward += 1
            
            self.in_range += 1
            
            if self.in_range > 50: 
                
                done = True 
                         
        if self.winner is 'None': 
            
            if (0 <= Att.x <= 2*MAP_CENTER[0] and 0 <= Att.y <= 2*MAP_CENTER[0] and Att.h.in_boundry()):

                done = False 

        if not Att.radar.TIM: 
            
            # some core search algorithms are not allowed to be made public            
            # target_Pos, target_h = recognization.probabilisticIndividual(Att.path)
            target_Pos, target_h = random_target.spherePos, random_target.h.real
        
        else: 
            
            target_Pos, target_h = Att.radar.TIM[0].spherePos, Att.radar.TIM[0].h.real
            
        dis, vec = util.displacement(Att.spherePos, target_Pos) # estimated
        
        disp = np.array([dis*vec[0]/15, dis*vec[1]/15, (target_h-Att.h.real) / 7.5]) # estimated                    
                                 
        Att.cESE(disp)
        
        Att.path.append(Att.spherePos)
                 
#        print(Att.pointState[-1], reward)
                                           
        return Att.pointState[-1], reward, done, angle_behavior
     
    #--------------------------------------------------------------------------
    def render(self):
        
        "环境展示"
        
        if self.viewer is None:
            
            self.viewer = Viewer(self.map, self.players, self.name)
            
        self.viewer.render(self.winner)
        
    #--------------------------------------------------------------------------
    def search_reward(self, player):
        
        """
        though we envisaged many ways to assess the quality of UCAV's movement 
        
        at a certain time, this look like the best one
        
        """
      
        min_max = [0, 1]
        
        l = player.v.upper*self.dt
        
        rx_ = -np.clip((player.spherePos[0]-7.5) / l+1, *min_max)
        
        ry_ = -np.clip((player.spherePos[1]-7.5) / l+1, *min_max)
        
        rh_ = -np.clip((player.h.real-8) / l+1, *min_max)
        
        _rx = -np.clip(1 - (player.spherePos[0]+7.5) / l, *min_max)
       
        _ry  = -np.clip(1 - (player.spherePos[1]+7.5) / l, *min_max)

        _rh  = -np.clip(1 - (player.h.real-0.5) / l, *min_max)
        
        r = sum([_rx, _ry, _rh, rx_, ry_, rh_])

        return r
    
#------------------------------------------------------------------------------
class benchAgent(object):
    
    def __init__(self, s_dim, a_dim):
    
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2+a_dim+1), dtype=np.float32)
            
        self.memory_full = False
        
        self.pointer = 0
        
    #--------------------------------------------------------------------------
    def store_transition(self, s, a, r, s_):
        
        None
        
#        transition = np.hstack((s, a, [r], s_))
#        
#        index = self.pointer % MEMORY_CAPACITY  
#        
#        self.memory[index, :] = transition
#        
#        self.pointer += 1
#        
#        if self.pointer > MEMORY_CAPACITY:     
#            
#            self.memory_full = True
            
    #--------------------------------------------------------------------------
    def choose_action(self, state=None):
        
        "generate motion"
                 
        action = npr.uniform(-1, 1, 2)  

        return action 
       
#------------------------------------------------------------------------------  
#------------------------------------------------------------------------------        
class Viewer(viewerUI.ViewerUI):
    
    "对抗过程可视化" 
    
    def __init__(self, _map, players, gameName):
        
        # vsync=False to speed up training
        super(Viewer, self).__init__(       
                                     width=1050, 
                                     height=600, 
                                     resizable=False, 
                                     caption=gameName, 
                                     vsync=False
        ) 
        
#        pyglet.gl.glClearColor(125, 125, 1, 1) 
        
        self.map = _map
        
        self.players = players
              
        for member in self.players['attackers']+self.players['defenders']:
            
            member.batch = self.batch
            
            member.group = self.foreground 
            
        self.map.batch, self.map.group = self.batch, self.background  
                
    #--------------------------------------------------------------------------
    def render(self, winner):
        
        "显示"
        
        self.update(winner)
        
        self.switch_to()
        
        self.dispatch_events()
        
        self.dispatch_event('on_draw')
        
        self.flip()
        
    #--------------------------------------------------------------------------
    def on_draw(self):
        
        self.clear()
        
        self.batch.draw()
        
    #--------------------------------------------------------------------------    
    def update(self, winner): 
        
        "更新显示界面" 
        
        # show for Operating cycle         
        self.OperatingStep += 1
        
        self.Operating_step_label.text = 'Operating Step: '+str(self.OperatingStep)
        
        self.winner_label.text = 'Winner: '+winner
        
        if self.OperatingStep%1 == 0:
        
            self.update_attributes(self.players['attackers'][0], 'attacker')  
    
            self.update_attributes(self.players['defenders'][0], 'defender')                  
                                                           
            for player in self.players['attackers']+self.players['defenders']: 
                
                player.update(self.dt) 

""                                                                           ""
"""----------------------------test fuction for EW env----------------------"""
""                                                                           ""                            
if __name__ == '__main__':    
    
    env = ElectronicWarfare_Env()

    MAX_EP_STEPS = env.levels(0)
    
    RENDER = 1  # rendering wastes time
    
    env.seed(1) 
    
    env.render()
    